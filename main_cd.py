from argparse import ArgumentParser
import torch
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
    from datasets.CD_dataset import CDDataset
    from torch.utils.data import DataLoader

    train_dataset = CDDataset(root_dir='datasets/SYSU-CD', img_size=args.img_size, split='train', is_train=True)
    val_dataset = CDDataset(root_dir='datasets/SYSU-CD', img_size=args.img_size, split='val', is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}

    from models.trainer import CDTrainer
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    from datasets.CD_dataset import CDDataset
    from torch.utils.data import DataLoader

    test_dataset = CDDataset(root_dir='datasets/SYSU-CD', img_size=args.img_size, split='test', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    from models.evaluator import CDEvaluator
    model = CDEvaluator(args=args, dataloader=test_loader)
    model.eval_models()

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='STENet', help='BIT_LEVIR | P2V_GZ | BIT_SYSU-CD | BIT_SYSU-CD', type=str)
    parser.add_argument('--checkpoint_root', default='yourpath/checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='SYSU-CD', type=str)

    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='STENet', type=str,
                        help='base_resnet18 | MYNet6 |'
                             'CICNet | BiFormer |P2VNet | TPNet | AFCF3D_Net')
    parser.add_argument('--loss', default='bce', help='bce | ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = r'E:\STENet\vis\STENet\predict'
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
