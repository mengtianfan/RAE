import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--models', help='target classifiers,Resnet50 or Inception_v3 or Densenet121',
                        default='Inception_v3')
    parser.add_argument('--eps', type=float,help=' budget', default=8/255)
    parser.add_argument('--inputpath', help='', default=r'/home/XMZ/val_randomselect')
    parser.add_argument('--outputpath', help='', default=r'./XMZ/recover_inception/')
    parser.add_argument('--pre_model', help='Init model', default=r'/home/aics/XMZ/new_model/model.pt')
    parser.add_argument('--pass_num', type=int, help='If you stop without finishing all images, you can restart at the stopped index', default=0)
    args = parser.parse_args()
    return args