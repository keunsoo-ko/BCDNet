import os, argparse
from torch.utils.data import DataLoader
from lib.utils import *
from lib.model import Net
from lib.data import Data
from tqdm import tqdm
from skimage.measure import compare_ssim
import time

parser = argparse.ArgumentParser(description="Official Pytorch Code for K. Ko et. al., Blind and Compact Denoising Network Based on Noise Order Learning, IEEE Trans. Image Process., vol. 31, pp. 1657-1670, Jan. 2022", usage='use "%(prog)s --help" for more information', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--data_path', required=True, help='Path for Test')
parser.add_argument('--model_path', required=True, help='Path for the pretrained model')

args = parser.parse_args()

ckpt_path = args.model_path

device = torch.device("cuda")

load_data = Data(args.data_path)
print('===> Loading datasets')

test_set = load_data.get_test_set()
data = DataLoader(dataset=test_set, num_workers=0,
                                 batch_size=1, shuffle=False)
models = Net().cuda()
models.load_state_dict(load_checkpoint(ckpt_path)['main'])
#models = nn.DataParallel(models).cuda()

params = sum(p.numel() for p in models.parameters())
print('===> Building model, Parameter #: ', params)

criterion = nn.MSELoss()
boundary_crop = 2

avg_psnr = 0
SSIM = 0.
len_data = len(data)
with torch.no_grad():
    for it, batch in enumerate(data):
        input, targets = \
            batch[0].to(device, dtype=torch.float),\
            batch[1].to(device, dtype=torch.float)
        
        feature = None
        CF = None
        output = input
        for i in range(3):
            output, _, feature, _, _, CF, _ = models(
                torch.cat([output, input-output], 1), torch.cat([output, input-output], 1), feature, feature, CF, CF)
        predict = output[..., boundary_crop:-boundary_crop, 
                                boundary_crop:-boundary_crop]
        target = targets[..., boundary_crop:-boundary_crop, 
                                boundary_crop:-boundary_crop]
        mse = criterion(predict, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
        pre = np.uint8(np.clip(predict[0].permute(1, 2, 0).cpu().numpy()*255., 0, 255))
        tar = np.uint8(np.clip(target[0].permute(1, 2, 0).cpu().numpy()*255., 0, 255))
        SSIM += compare_ssim(pre, tar, multichannel=True)
        
        #cv2.imwrite("results/%d_predict.png"%(it), pre)
        #cv2.imwrite("results/%d_targets.png"%(it), tar)
        print("{}: PSNR {} / SSIM {}".format(it, avg_psnr / (it+1), SSIM / (it+1)))
print("total: ", avg_psnr / len_data, SSIM / len_data)
