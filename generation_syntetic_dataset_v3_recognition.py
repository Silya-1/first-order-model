import imageio
import matplotlib.animation as animation
import face_recognition
import face_alignment
import numpy as np
import torch
import glob
import os
import sys
import warnings

from IPython.display import HTML
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from scipy.spatial import ConvexHull
from demo import load_checkpoints, make_animation, find_best_frame
from scipy import linalg


PATH_TO_FOM = '/content/first-order-model'
PATH_TO_IMG_DIR = '/content/gdrive/My Drive/dl/celeb_hq/256x256/data256x256/'
PATH_TO_VIDEO_DIR = '/content/gdrive/My Drive/dl/video_celeb/'
CHECKPOINT_PATH = '/content/gdrive/My Drive/first-order-motion-model/vox-cpk.pth.tar'
CONFIG_PATH = '/content/first-order-model/config/vox-256.yaml'
NEW_DATASET_PATH = '/content/gdrive/My Drive/dl/celeb_hq/test_new_celeb'

sys.path.append(PATH_TO_FOM)


def main():

    step = 15
    N_total_images = 2
    folders = ['source', 'drive', 'predict']

    dataset_step_mode = 'grid'
    grid_step = 2
    # dataset_step_mode = 'most_distant'
    # n_frames = 10 # define if  dataset_step_mode == 'most_distant'

    face_comparison_mode = 'keypoints'
    # face_comparison_mode = 'recognition_net'

    np.random.seed(0)

    for folder in folders:
        remove(folder, NEW_DATASET_PATH)

    generator, kp_detector = load_checkpoints(config_path=CONFIG_PATH,
                                              checkpoint_path=CHECKPOINT_PATH)
    path_getter = lambda x: list(map(lambda y: os.path.join(os.path.abspath(x), y),
                                                            os.listdir(x)))

    total = 0
    for n_video, video_path in enumerate(path_getter(PATH_TO_VIDEO_DIR)):
        if total >= N_total_images:
            break
            
        driving_video = imageio.mimread(video_path, memtest=False)
        driving_video = [resize(frame, (256, 256))[..., :3]
                                 for frame in driving_video]
        
        for n_source, source_path in enumerate(path_getter(PATH_TO_IMG_DIR)):
            if total >= N_total_images:
                break
        
            source_image = imageio.imread(source_path)
            source_image = resize(source_image, (256, 256))[..., :3]
            
            ### ids - indexes of frames that we will use for dataset making, every "step" frame###
            ids = np.arange(0, len(driving_video), step)
            driving_video = np.array(driving_video)[ids]

            ### id of closest frame by keypoints
            try:
                norms_for_best_source = get_key_points(source_image, driving_video)
            except (RuntimeError, TypeError):
                print('Image not processed by recognition, moving to next image')
                continue
            best_i = np.argmin(norms_for_best_source)
            
            swap = driving_video[0]
            driving_video[0] = driving_video[best_i]
            driving_video[best_i] = swap
            
            predictions = make_animation(source_image, driving_video,
                                         generator, kp_detector, relative=True,
                                         adapt_movement_scale=True)
            
            
            ### apply face comparison
            if face_comparison_mode == 'keypoints':
                try:
                    norms_for_best_source = get_key_points(source_image, predictions)
                except (RuntimeError, TypeError, NameError):
                    print('Image not processed by get_key_points, moving to next image')
                    continue
                    
            elif face_comparison_mode == 'recognition_net':
                try:
                    norms_for_best_source = recognition(source_image, predictions)
                except (RuntimeError, TypeError, NameError):
                    print('Image not processed by recognition_net, moving to next image')
                    continue
            
            ids_for_best_preds = np.argsort(norms_for_best_source)[::-1]
            
            ### set the range for different modes
            if dataset_step_mode == 'grid':
                frame_range = ids_for_best_preds[: len(ids_for_best_preds) : grid_step]
            elif dataset_step_mode == 'most_distant':
                frame_range = ids_for_best_preds[: min(n_frames, len(ids_for_best_preds))]
            else:
                print('Unknown dataset_step_mode')
                break
                
                
            for id in frame_range:
                drive = driving_video[id]
                pred = predictions[id]
                triplet = np.stack([source_image[None,:,:,:],
                                  drive[None,:,:,:], pred[None,:,:,:]], axis=0)
                name = '_'.join([str(n_source), str(n_video), str(id)])
                for i, folder in enumerate(folders):
                    save(name, NEW_DATASET_PATH, folder, triplet[i])
                total += 1 # if accept
                if total >= N_total_images:
                    break
                
                
                
def remove(name_folder, path=NEW_DATASET_PATH):
   for f in glob.glob(os.path.join(NEW_DATASET_PATH, name_folder, "*")):
     os.remove(f)

def save(name, root_path, name_folder, data):
  path = os.path.join(root_path, name_folder, name)
  im = Image.fromarray(np.uint8(data.reshape(data.shape[1:])*255))
  im.save(path +'.png')

  
  
def recognition(source_image, driving_frames):
    source_image_encoding = face_recognition.face_encodings(np.array(source_image*256).astype(np.uint8))[0]
    driving_encoding = [face_recognition.face_encodings(np.array(frame*256).astype(np.uint8))[0] for frame in driving_frames]
    face_distances = face_recognition.face_distance(driving_encoding, source_image_encoding)
    return(face_distances)
  

def get_key_points(source, driving, cpu=False):

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norms  = []
    for i, image in enumerate(driving):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        norms.append((np.abs(kp_source - kp_driving) ** 2).sum())
    
    return np.array(norms)


if __name__ == '__main__':
    main()
