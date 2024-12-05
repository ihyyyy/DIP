

from draggan.api import drag_gan, stylegan2
from draggan.stylegan2.inversion import inverse_image
from draggan import utils
from PIL import Image
import torch


device = 'cuda'
CKPT_SIZE = {
    'stylegan2-ffhq-config-f.pt': 1024,
    'stylegan2-cat-config-f.pt': 256,
    'stylegan2-church-config-f.pt': 256,
    'stylegan2-horse-config-f.pt': 256,
    'ada/ffhq.pt': 1024,
    'ada/afhqcat.pt': 512,
    'ada/afhqdog.pt': 512,
    'ada/afhqwild.pt': 512,
    'ada/brecahad.pt': 512,
    'ada/metfaces.pt': 512,
    'human/v2_512.pt': 512,
    'human/v2_1024.pt': 1024,
    'self_distill/bicycles_256.pt': 256,
    'self_distill/dogs_1024.pt': 1024,
    'self_distill/elephants_512.pt': 512,
    'self_distill/giraffes_512.pt': 512,
    'self_distill/horses_256.pt': 256,
    'self_distill/lions_512.pt': 512,
    'self_distill/parrots_512.pt': 512,
}

#人脸数据集
DEFAULT_CKPT = 'stylegan2-ffhq-config-f.pt'

class ModelWrapper:
    def __init__(self, **kwargs):
        self.g_ema = stylegan2(**kwargs).to(device)


def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')




def on_image_change(model, image_size, image):
    image = Image.fromarray(image)
    result = inverse_image(
        model.g_ema,
        image,
        image_size=image_size
    )
    result['history'] = []
    image = to_image(result['sample'])
    points = {'target': [], 'handle': []}
    target_point = False
    return image, image, result, points, target_point


wrapped_model = ModelWrapper(ckpt=DEFAULT_CKPT, size=CKPT_SIZE[DEFAULT_CKPT])
g_ema = wrapped_model.g_ema

import numpy as np
import cv2
#load the image ycz.jpg and convert it to numpy array and compute its size
image = Image.open('./ycz.jpg')
image = np.array(image)



image, mask, state, points, target_point=on_image_change(wrapped_model, CKPT_SIZE[DEFAULT_CKPT], image)

# 将numpy array转换为OpenCV格式
img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite('ycz_inverted.jpg', img)

#save image mask state to files
# 保存多个数组 (npz格式)
np.savez('arrays.npz', image=image, mask=mask)
# # 读取
# loaded_arrays = np.load('arrays.npz')
# image = loaded_arrays['image']
# mask = loaded_arrays['mask']

import pickle
# 写入
with open('state.pkl', 'wb') as f:
    pickle.dump(state, f)
# # 读取
# with open('state.pkl', 'rb') as f:
#     loaded_state = pickle.load(f)
