
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 4
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]


batch_size = 16
cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5


cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 50



TRAIN_PATH = '/home/xmz/Dataset/DIV2K/DIV2K_train_HR/'
VAL_PATH = '/home/xmz/Dataset/DIV2K/DIV2K_valid_HR/'
format_train = 'png'
format_val = 'png'


loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False




MODEL_PATH = '/home/xmz/model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/home/xmz/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'cover-rev/'


suffix = '/home/xmz/model.pt'
tain_next = False
trained_epoch = 0
