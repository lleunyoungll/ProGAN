""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config

torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
            
        ]
    )
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan1")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    if config.MODE=="train":
        gen.train()
        critic.train()

        tensorboard_step = 0
        # start at step that corresponds to img size that we set in config
        step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
        #step=512
        for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
            alpha = 1e-5  # start with very low alpha
            loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
            print(f"Current image size: {4 * 2 ** step}")

            for epoch in range(num_epochs):
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                tensorboard_step, alpha = train_fn(
                    critic,
                    gen,
                    loader,
                    dataset,
                    step,
                    alpha,
                    opt_critic,
                    opt_gen,
                    tensorboard_step,
                    writer,
                    scaler_gen,
                    scaler_critic,
                )

                if config.SAVE_MODEL:
                    save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                    save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)
                    if epoch%100==0:
                        save_checkpoint(gen, opt_gen, filename="generator_epoch"+str(epoch+1)+"_size"+str(4 * 2 ** step)+".pth")
                        save_checkpoint(critic, opt_critic, filename="critic_epoch"+str(epoch+1)+"_size"+str(4 * 2 ** step)+".pth")
                    generate_examples(gen,step)

            step += 1  # progress to the next img size
    else:
        step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step}")
        loop = tqdm(loader, leave=True)
        for batch_idx, (real, _) in enumerate(loop):
            #save_image(real, f"realfakeRes/"+str(batch_idx)+"real_original.png")
            real = real.to(config.DEVICE)
            cur_batch_size = real.shape[0]
            print(cur_batch_size)
            alpha = 1e-5  # start with very low alpha
            alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
            )
            alpha = min(alpha, 1)
            print("alpha: "+str(alpha))
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step)
                #fixed_fakes = gen(real.detach(), alpha, step)
                
                #fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            #realImg = torchvision.utils.make_grid(real.detach()[:8], normalize=True)
            realImg=real.detach()

            fakeImg=fixed_fakes.detach()
            img_grid_fake = torchvision.utils.make_grid(fakeImg[:8], normalize=True)
            save_image(realImg, f"realfakeRes/"+str(batch_idx)+"real.png")
            save_image(img_grid_fake, f"realfakeRes/"+str(batch_idx)+"fake.png")



    


if __name__ == "__main__":
    main()
