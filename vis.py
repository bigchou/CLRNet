import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse, pdb
#from lanedet.datasets.process import Process
#from lanedet.models.registry import build_net
#from lanedet.utils.config import Config
#from lanedet.utils.visualization import imshow_lanes
#from lanedet.utils.net_utils import load_network

"""
Problem solved by using the lanedet/tools/detect.py

Need to change:

1. import library
2. line 38: data = self.net.module.heads.get_lanes(data)
3. config: width, height, cut_height
Thanks to #46
"""

# https://github.com/Turoad/CLRNet/issues/90
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network


from pathlib import Path
from tqdm import tqdm

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path) # (1080, 1920, 3)

        ori_img = cv2.resize(ori_img, (self.cfg.ori_img_w, self.cfg.ori_img_h)) # <--------------
        # ori_img.shape = (717, 1276, 3)

        img = ori_img[self.cfg.cut_height:, :, :]
        # self.cfg.cut_height = 420

        # img.shape = (297, 1276, 3)
        img = img.astype(np.float32)
        #img = ori_img.astype(np.float32)
        #pdb.set_trace()




        data = {'img': img, 'lanes': []}
        # cv2.imwrite('qasd.png', img.astype(np.uint8))
        #import pdb
        #pdb.set_trace()
        print(img.shape)
        data = self.processes(data)
        print(data['img'].shape)
        #pdb.set_trace()
        data['img'] = data['img'].unsqueeze(0)
        
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            # https://github.com/Turoad/CLRNet/issues/44
            #data = self.net.module.get_lanes(data)
            data = self.net.module.heads.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        print(out_file)
        print("#lanes:",len(lanes), [lane.shape for lane in lanes])
        #print("#lanes:", len(lanes), lanes[0].shape, lanes[1].shape, lanes[2].shape, lanes[3].shape)
        print("imgpath: ", data['img_path'])




        # draw mask
        #pdb.set_trace()
        
        lanes_xys = []
        for _, lane in enumerate(lanes):
            xys = []
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                xys.append((x, y))
            lanes_xys.append(xys)
        lanes_xys.sort(key=lambda xys : xys[0][0])
        width = 4
        mask = np.zeros( (int(self.cfg.ori_img_h), int(self.cfg.ori_img_w)) ).astype(np.uint8)
        print("mask.shape:", mask.shape)
        for idx, xys in enumerate(lanes_xys):
            for i in range(1, len(xys)):
                print(idx+1)
                #pdb.set_trace()
                cv2.line(mask, xys[i - 1], xys[i], int(idx+1), thickness=width)
            print("----")
        
        #pdb.set_trace()
        img = cv2.imread(data['img_path'])
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg = np.zeros(img.shape).astype(np.uint8)
        uni = np.unique(mask)
        #pdb.set_trace()
        colors = [
            None,
            [255, 0, 0], # blue
            [0, 255, 0], # green
            [0, 0, 255], # red
            [255, 255, 0],
            [0, 255, 255]
        ]
        
        for i in uni:
            if i == 0:
                continue
            seg[mask==i] = colors[i]
        segim = ((seg.astype(np.float32) + img.astype(np.float32))//2).astype(np.uint8)
        print(img.shape, mask.shape, seg.shape, segim.shape)
        
        
        if out_file:
            if not osp.exists(osp.dirname(out_file)):
                os.makedirs(osp.dirname(out_file))
            cv2.imwrite(out_file, segim)
            print('save to ', out_file)
        
        
        
        
        
        
        #pdb.set_trace()
        #imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        print("END...")
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        print("xxxxxx")
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    print('paths:', paths)
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)


# python vis.py configs/clrnet/clr_resnet34_culane.py --img 05081544_0305-001868.jpg --load_from culane_r34.pth --savedir ./output
# input image size is 590height 1640width

# (openmmlab) iis@iis-Z590-AORUS-ELITE-AX:~/Desktop/CLRNet$ python vis.py configs/clrnet/clr_dla34_llamas_mod.py --img images --load_from llamas_dla34.pth --savedir ./output
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
"""
(openmmlab) iis@iis-Z590-AORUS-ELITE-AX:~/Desktop/CLRNet$ pip list
Package            Version     Location
------------------ ----------- ------------------------
addict             2.4.0
albumentations     0.4.6
brotlipy           0.7.0
certifi            2022.5.18.1
cffi               1.15.0
charset-normalizer 2.0.4
click              7.1.2
clrnet             1.0         /home/iis/Desktop/CLRNet
colorama           0.4.5
cryptography       37.0.1
cycler             0.11.0
dill               0.3.7
fonttools          4.33.3
idna               3.3
imageio            2.33.0
imgaug             0.4.0
importlib-metadata 4.11.4
joblib             1.3.2
kiwisolver         1.4.3
lazy_loader        0.3
Markdown           3.3.7
matplotlib         3.5.2
mkl-fft            1.3.1
mkl-random         1.2.2
mkl-service        2.4.0
mmcv               1.2.5
mmcv-full          1.5.0
model-index        0.1.11
multiprocess       0.70.15
networkx           3.1
numpy              1.22.3
opencv-python      4.6.0.66
openmim            0.1.6
ordered-set        4.1.0
p-tqdm             1.4.0
packaging          21.3
pandas             1.4.2
pathos             0.3.1
pathspec           0.11.2
Pillow             9.0.1
pip                21.2.4
pox                0.3.3
ppft               1.7.6.7
ptflops            0.7.1.2
pycocotools        2.0.4
pycparser          2.21
pyOpenSSL          22.0.0
pyparsing          3.0.9
PySocks            1.7.1
python-dateutil    2.8.2
pytz               2022.1
PyWavelets         1.4.1
PyYAML             6.0
requests           2.27.1
scikit-image       0.21.0
scikit-learn       1.3.2
scipy              1.10.1
setuptools         61.2.0
Shapely            1.7.0
six                1.16.0
sklearn            0.0
tabulate           0.8.9
terminaltables     3.1.10
threadpoolctl      3.2.0
tifffile           2023.7.10
timm               0.9.11
torch              1.11.0
torchvision        0.12.0
tqdm               4.66.1
typing_extensions  4.1.1
ujson              1.35
urllib3            1.26.9
wheel              0.37.1
yapf               0.32.0
zipp               3.8.0
"""

"""
(openmmlab) iis@iis-Z590-AORUS-ELITE-AX:~/Desktop/CLRNet$ conda list
# packages in environment at /home/iis/anaconda3/envs/openmmlab:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
addict                    2.4.0                    pypi_0    pypi
blas                      1.0                         mkl  
brotlipy                  0.7.0           py38h27cfd23_1003  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2022.4.26            h06a4308_0  
certifi                   2022.5.18.1      py38h06a4308_0  
cffi                      1.15.0           py38hd667e15_1  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     7.1.2                    pypi_0    pypi
clrnet                    1.0                       dev_0    <develop>
colorama                  0.4.5                    pypi_0    pypi
cryptography              37.0.1           py38h9ce1e76_0  
cudatoolkit               11.3.1               h2bc3f7f_2  
cycler                    0.11.0                   pypi_0    pypi
dill                      0.3.7                    pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
fonttools                 4.33.3                   pypi_0    pypi
freetype                  2.11.0               h70c0345_0  
giflib                    5.2.1                h7b6447c_0  
gmp                       6.2.1                h295c915_3  
gnutls                    3.6.15               he1e5248_0  
idna                      3.3                pyhd3eb1b0_0  
imageio                   2.33.0                   pypi_0    pypi
imgaug                    0.4.0                    pypi_0    pypi
importlib-metadata        4.11.4                   pypi_0    pypi
intel-openmp              2021.4.0          h06a4308_3561  
joblib                    1.3.2                    pypi_0    pypi
jpeg                      9e                   h7f8727e_0  
kiwisolver                1.4.3                    pypi_0    pypi
lame                      3.100                h7b6447c_0  
lazy-loader               0.3                      pypi_0    pypi
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h7f8727e_2  
libidn2                   2.3.2                h7f8727e_0  
libpng                    1.6.37               hbc83047_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtasn1                  4.16.0               h27cfd23_0  
libtiff                   4.2.0                h2818925_1  
libunistring              0.9.10               h27cfd23_0  
libuv                     1.40.0               h7b6447c_0  
libwebp                   1.2.2                h55f646e_0  
libwebp-base              1.2.2                h7f8727e_0  
lz4-c                     1.9.3                h295c915_1  
markdown                  3.3.7                    pypi_0    pypi
matplotlib                3.5.2                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py38h7f8727e_0  
mkl_fft                   1.3.1            py38hd3c417c_0  
mkl_random                1.2.2            py38h51133e4_0  
mmcv                      1.2.5                    pypi_0    pypi
mmcv-full                 1.5.0                    pypi_0    pypi
model-index               0.1.11                   pypi_0    pypi
multiprocess              0.70.15                  pypi_0    pypi
ncurses                   6.3                  h7f8727e_2  
nettle                    3.7.3                hbbd107a_1  
networkx                  3.1                      pypi_0    pypi
numpy                     1.22.3           py38he7a7128_0  
numpy-base                1.22.3           py38hf524024_0  
opencv-python             4.6.0.66                 pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openmim                   0.1.6                    pypi_0    pypi
openssl                   1.1.1o               h7f8727e_0  
ordered-set               4.1.0                    pypi_0    pypi
packaging                 21.3                     pypi_0    pypi
pandas                    1.4.2                    pypi_0    pypi
pathos                    0.3.1                    pypi_0    pypi
pathspec                  0.11.2                   pypi_0    pypi
pillow                    9.0.1            py38h22f2fdc_0  
pip                       21.2.4           py38h06a4308_0  
pox                       0.3.3                    pypi_0    pypi
ppft                      1.7.6.7                  pypi_0    pypi
pycocotools               2.0.4                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0  
pyopenssl                 22.0.0             pyhd3eb1b0_0  
pyparsing                 3.0.9                    pypi_0    pypi
pysocks                   1.7.1            py38h06a4308_0  
python                    3.8.13               h12debd9_0  
python-dateutil           2.8.2                    pypi_0    pypi
pytorch                   1.11.0          py3.8_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.1                   pypi_0    pypi
pywavelets                1.4.1                    pypi_0    pypi
pyyaml                    6.0                      pypi_0    pypi
readline                  8.1.2                h7f8727e_1  
requests                  2.27.1             pyhd3eb1b0_0  
scikit-image              0.21.0                   pypi_0    pypi
scikit-learn              1.3.2                    pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
setuptools                61.2.0           py38h06a4308_0  
shapely                   1.7.0                    pypi_0    pypi
six                       1.16.0             pyhd3eb1b0_1  
sklearn                   0.0                      pypi_0    pypi
sqlite                    3.38.3               hc218d9a_0  
tabulate                  0.8.9                    pypi_0    pypi
terminaltables            3.1.10                   pypi_0    pypi
threadpoolctl             3.2.0                    pypi_0    pypi
tifffile                  2023.7.10                pypi_0    pypi
timm                      0.9.11                   pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
torchvision               0.12.0               py38_cu113    pytorch
tqdm                      4.66.1                   pypi_0    pypi
typing_extensions         4.1.1              pyh06a4308_0  
ujson                     1.35                     pypi_0    pypi
urllib3                   1.26.9           py38h06a4308_0  
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.2.5                h7f8727e_1  
yapf                      0.32.0                   pypi_0    pypi
zipp                      3.8.0                    pypi_0    pypi
zlib                      1.2.12               h7f8727e_2  
zstd                      1.5.2                ha4553b6_0
"""
