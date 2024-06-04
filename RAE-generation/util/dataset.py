from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from util import choose_target
from args import get_args_parser
from util.utils import *
args = get_args_parser()

class Dataset(Dataset):
    def __init__(self, transforms_=None):

        self.transform = transforms_
        self.TRAIN_PATH = args.inputpath
        self.format_train = 'png'
        self.files = natsorted(sorted(imglist(self.TRAIN_PATH, self.format_train)))

    def __getitem__(self, index):
            image = Image.open(self.files[index+args.pass_num])
            image = to_rgb(image)
            item = self.transform(image)
            item = item.unsqueeze(0)
            filename = self.files[index+args.pass_num].split("/")
            classname = filename[len(filename)-2]
            classindex = cindex(classname)
            targetclass = choose_target.choose_target(classname)
            tarindex = cindex(targetclass)
            return item, classindex,tarindex
    def __len__(self):
        return len(self.files)

transform = T.Compose([
    T.ToTensor(),
])

trainloader = DataLoader(
    Dataset(transforms_=transform),
    batch_size=1,
    shuffle=False,
    pin_memory=False,
    num_workers=args.workers,
    drop_last=True
)

