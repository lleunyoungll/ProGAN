import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 1024
DATASET = 'data_validation_other'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL=True
MODE="eval"
LEARNING_RATE = 1e-3 #1e-3
IMAGE_SIZE=1024
#BATCH_SIZES = [32, 32, 32, 32, 32, 32, 32, 32, 32]
#BATCH_SIZES = [4, 4, 4, 4, 4, 4, 4, 4, 4]
BATCH_SIZES = [1, 1,1,1,1,1,1,1,1]
#BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [60000] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(1, Z_DIM, 1, 1).to(DEVICE) #torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4