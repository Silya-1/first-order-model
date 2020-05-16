PATH_TO_IMG_DIR = 'content/imgs/data256x256/'
PATH_TO_VIDEO_DIR = 'content/vids/'
CHECKPOINT_PATH = 'content/vox-cpk.pth.tar'
CONFIG_PATH = 'config/vox-256.yaml'
NEW_DATASET_PATH = '../gan-compression/datasets/fomm/'

N_images_orig = 1000
N_videos = 5
N_images_from_video = 10
folders = ['source', 'drive', 'predict']

import os
import sys
import glob
import warnings
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from PIL import Image  
import imageio
from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
np.random.seed(0)

from demo import load_checkpoints, make_animation




def remove(name_folder, path=NEW_DATASET_PATH):
    for f in glob.glob(os.path.join(NEW_DATASET_PATH, name_folder, "*")):
        os.remove(f)

def save(name, root_path, name_folder, data):
    path = os.path.join(root_path, name_folder, name)
    im = Image.fromarray(np.uint8(data.reshape(data.shape[1:])*255))
    im.save(path +'.png')


for folder in folders:
    remove(folder, NEW_DATASET_PATH)

generator, kp_detector = load_checkpoints(config_path=CONFIG_PATH, 
                                          checkpoint_path=CHECKPOINT_PATH)

path_getter = lambda x: list(map(lambda y: os.path.join(os.path.abspath(x), y), 
                                                        os.listdir(x)))


for n_source, source_path in tqdm(enumerate(path_getter(PATH_TO_IMG_DIR)),
                                  total=N_images_orig):
    if not n_source < N_images_orig:
        break
    for n_video, video_path in enumerate(path_getter(PATH_TO_VIDEO_DIR)):
        if not n_video < N_videos:
            break
        source_image = imageio.imread(source_path)
        driving_video = imageio.mimread(video_path, memtest=False)

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

      
        ids = np.random.choice(len(driving_video), N_images_from_video-1, replace=False)
        driving_video = np.array(driving_video)[np.hstack(([0], ids))]
        
        predictions = make_animation(source_image, driving_video,
                                     generator, kp_detector, relative=True,
                                     adapt_movement_scale=True)
        
        for id in range(len(ids)):
            drive = driving_video[id]
            pred = predictions[id]
            triplet = np.stack([source_image[None,:,:,:], 
                          drive[None,:,:,:], pred[None,:,:,:]], axis=0)
            name = '_'.join([str(n_source), str(n_video), str(id)])
        
        for i, folder in enumerate(folders):
            save(name, NEW_DATASET_PATH, folder, triplet[i])