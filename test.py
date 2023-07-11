import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.BAImodel import BAINet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./BAI_dataset/RGBD_for_test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = BAINet()

model.load_state_dict(torch.load(('./BAINet_epoch_best.pth'),map_location='cpu'))

model.cuda()
model.eval()

#test
# test_datasets = ['NJU2K','NLPR','STERE', 'DES','SIP']
test_datasets = ['DES']
for dataset in test_datasets:
    save_path = './test_maps/BAINet_salmap/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    mae_sum=0
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        _,res = model(image,depth)
        res,res1,res2,res3,res4,res5,res6,res7=model(image, depth)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])#新加

        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)

    mae = mae_sum / test_loader.size
    print("MAE",mae)
    print('Test Done!')
