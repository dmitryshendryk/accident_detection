BATCH_SIZE=10
BN_SIZE=4  #Expansion rate for Bottleneck structure.
GROWTH_RATE=32 #Fixed number of out_channels for some layers.
KEEP_PROB=1.0  # 1-dropout rate
START_CHANNEL=64  #number of channels before entering first block structure.
IS_TRAIN=True   #Must be true when training.
CROP_SIZE=160   #height and width of input video clip 
NUM_FRAMES_PER_CLIP=16  #length for each clip
NUM_CLASSES=2
IS_DA=False #whether or not to use data augmentation
