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


def add_points_to_image(image, points, size=5):
    image = utils.draw_handle_target_points(image, points['handle'], points['target'], size)
    return image


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



def on_drag(model, points, max_iters, state, size, mask):
    max_iters = int(max_iters)
    latent = state['latent']
    noise = state['noise']
    F = state['F']

    handle_points = [torch.tensor(p, device=device).float() for p in points['handle']]
    target_points = [torch.tensor(p, device=device).float() for p in points['target']]

    if mask is not None:
        mask = Image.fromarray(mask).convert('L')
        mask = np.array(mask) == 255

        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None

    step = 0
    for sample2, latent, F, handle_points in drag_gan(model.g_ema, latent, noise, F,
                                                      handle_points, target_points, mask,
                                                      max_iters=max_iters):
        image = to_image(sample2)

        state['F'] = F
        state['latent'] = latent
        state['sample'] = sample2
        # points['handle'] = [p.cpu().numpy().astype('int') for p in handle_points]
        # image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])

        state['history'].append(image)
        step += 1
        
        # print(handle_points)
        # print(target_points)
        yield image, state, step


      


# load the image ycz_inverted.jpg and convert to numpy array

import numpy as np
import cv2

preds=np.load('preds.npy')
points = {'target': [], 'handle': []}

for i in [48,54,57]:
    points['handle'].append([preds[0,i,1],preds[0,i,0]])


image = Image.open('./ycz_inverted.jpg')
image = np.array(image)

img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for i in [48,54,57]:
    cv2.circle(img, (int(preds[0,i,0]), int(preds[0,i,1])), 1, (0,0,255), -1)


# preds point coordinates (x,y), from top-left to bottom-right
# draggan point coordinates (y,x), from top-left to bottom-right


d=13
preds[0,48,:]+=[-d,-d]
preds[0,54,:]+=[d,-d]
preds[0,57,:]+=[0,d+5]



for i in [48,54,57]:
    points['target'].append([preds[0,i,1],preds[0,i,0]])

target_point = False
for i in [48,54,57]:
    cv2.circle(img, (int(preds[0,i,0]), int(preds[0,i,1])), 1, (0,0,255), -1)


cv2.imwrite('test2.jpg', img)

# state = gr.State({
#     'latent': latent,
#     'noise': noise,
#     'F': F,
#     'sample': sample,
#     'history': []
# })


loaded_arrays = np.load('arrays.npz')
image = loaded_arrays['image']
mask = loaded_arrays['mask']

Image.fromarray(image).save('image.jpg')
Image.fromarray(mask).save('mask.jpg')
import pickle

# 读取
with open('state.pkl', 'rb') as f:
    state = pickle.load(f)



max_iters=50
model = ModelWrapper(ckpt=DEFAULT_CKPT, size=CKPT_SIZE[DEFAULT_CKPT])

size=CKPT_SIZE[DEFAULT_CKPT]
mask = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)


for image, state, progress in on_drag(model, points, max_iters, state, size, mask):
    print('iteration:', progress)


Image.fromarray(image).save('smile_face.jpg')





