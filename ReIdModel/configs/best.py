import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



P = 64 
K = 4  

NUM_POINTS = 5000 

NUM_WORKERS = 4


FEATURE_DIM = 6
EMBEDDING_DIM = 512


LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4

LR_SCHEDULER_STEP = 40
LR_SCHEDULER_GAMMA = 0.1

EPOCHS = 200

LOSS_MARGIN = 0.05


LOG_PERIOD = 50