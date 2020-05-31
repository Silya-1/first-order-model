import argparse

from demo import load_checkpoints, make_animation, find_best_frame

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.autograd import Variable

from skimage import img_as_ubyte
from skimage.transform import resize
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import HTML
from PIL import Image
import imageio

import face_recognition
import face_alignment

from scipy.spatial import ConvexHull
from scipy import linalg

import sys
import os
import glob

import warnings
warnings.filterwarnings("ignore")

total_dirs = 3
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Process neccesary args')
parser.add_argument('--fomm_path', action='store', dest='fomm_path', type=str,
                    default='/content/first-order-model',
                    help='Path to fomm')

parser.add_argument('--img_path', action='store', dest='img_path', type=str,
                    default='/content/gdrive/My Drive/dl/celeb_hq/256x256/data256x256/',
                    help='Path to image dir')

parser.add_argument('--chekpoint_path', action='store', dest='chekpoint_path', type=str,
                    default='/content/gdrive/My Drive/first-order-motion-model/vox-cpk.pth.tar',
                    help='Path to checkpoint')

parser.add_argument('--video_path', action='store', dest='video_path', type=str,
                    default='/content/gdrive/My Drive/dl/video_celeb/',
                    help='Path to video folder')

parser.add_argument('--config_path', action='store', dest='config_path', type=str,
                    default='/content/first-order-model/config/vox-256.yaml',
                    help='Path to config folder')

parser.add_argument('--new_dataset_path', action='store', dest='new_dataset_path', type=str,
                    default='/content/gdrive/My Drive/dl/celeb_hq/test_new_celeb',
                    help='Path to new dataset folder')

parser.add_argument('--step_video', action='store', dest='step_video', type=int,
                    default=30,
                    help='Step in video grid')

parser.add_argument('--dataset_step_mode', action='store', dest='dataset_step_mode', type=str,
                    default='grid',
                    help='Mode to choose frames')

parser.add_argument('--grid_step', action='store', dest='grid_step', type=int,
                    default=2,
                    help='Grid step')

parser.add_argument('--n_frames', action='store', dest='n_frames', type=int,
                    default=5,
                    help='Number of most distant frames')

parser.add_argument('--face_comparison_mode', action='store', dest='face_comparison_mode', type=str,
                    default='recognition',
                    help='Mode to select distance')

parser.add_argument('--N_total_images', action='store', dest='N_total_images', type=int,
                    default=10,
                    help='Number of total images to generate')

parser.add_argument('--folders_to_store', action='store', dest='folders_to_store', type=list,
                    default=['source', 'drive', 'predict'],
                    help='Name of folders for new dataset')

parser.add_argument('--if_clean_folders', action='store', dest='if_clean_folders', type=bool,
                    default=True,
                    help='Flag to clear folders')


def clean_one_folder(folder_name, folder_path):
   for f in glob.glob(os.path.join(folder_path, folder_name, "*")):
        os.remove(f)


def clean_folders(folders, folder_path):
    for folder in folders:
        clean_one_folder(folder, folder_path)


def save(name, root_path, folder_name, data):
    path = os.path.join(root_path, folder_name, name)
    im = Image.fromarray(np.uint8(data.reshape(data.shape[1:])*255))
    im.save(path + '.png')


def create_folders(folders, folder_path):
    for folder in folders:
        full_folder_path = os.path.join(folder_path, folder)
        if not os.path.isdir(full_folder_path):
            os.makedirs(full_folder_path)


def path_getter(x): return list(map(lambda y: os.path.join(os.path.abspath(x), y),
                                    os.listdir(x)))


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
    norms = []
    for i, image in enumerate(driving):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        norms.append((np.abs(kp_source - kp_driving) ** 2).sum())
    return np.array(norms)


def recognition(source_image, driving_frames):
    source_image_encoding = face_recognition.face_encodings(
        np.array(source_image*256).astype(np.uint8))[0]
    driving_encoding = [face_recognition.face_encodings(
        np.array(frame*256).astype(np.uint8))[0] for frame in driving_frames]
    face_distances = face_recognition.face_distance(
        driving_encoding, source_image_encoding)
    return(face_distances)


def generate_data(generator, kp_detector, N_total_images, step_video,
                  video_path, img_path, new_dataset_path, folders, face_comparison_mode,
                  dataset_step_mode, n_frames, grid_step):
    total = 0
    for n_video, video_path in enumerate(path_getter(video_path)):
        if total >= N_total_images:
            break
        driving_video = imageio.mimread(video_path, memtest=False)
        driving_video = [resize(frame, (256, 256))[..., :3]
                         for frame in driving_video]

        ids = np.arange(0, len(driving_video), step_video)
        driving_video = np.array(driving_video)[ids]

        for n_source, source_path in enumerate(path_getter(img_path)):
                if total >= N_total_images:
                    break
                source_image = imageio.imread(source_path)
                source_image = resize(source_image, (256, 256))[..., :3]

                try:
                    norms_for_best_source = get_key_points(
                        source_image, driving_video)
                except Exception:
                    print(
                        'Image not processed by get_key_points, moving to next image')
                    continue

                best_i = np.argmin(norms_for_best_source)
                driving_video[0], driving_video[best_i] = driving_video[best_i], driving_video[0]

                predictions = make_animation(source_image, driving_video,
                                             generator, kp_detector, relative=True,
                                             adapt_movement_scale=True)

                if face_comparison_mode == 'keypoints':
                    try:
                        norms_for_best_preds = get_key_points(
                            source_image, predictions)
                    except Exception:
                        print(
                            'Image not processed by get_key_points, moving to next image')
                        continue

                elif face_comparison_mode == 'recognition':
                    try:
                        norms_for_best_preds = recognition(
                            source_image, predictions)
                    except Exception:
                        print(
                            'Image not processed by recognition_net, moving to next image')
                        continue
                else:
                    print("Uknown parameter")
                    sys.exit(-1)

                ids_for_best_preds = np.argsort(norms_for_best_preds)[::-1]

                if dataset_step_mode == 'grid':
                    ids_for_best_preds = ids_for_best_preds[: len(
                        ids_for_best_preds): grid_step]
                elif dataset_step_mode == 'most_distant':
                    ids_for_best_preds = ids_for_best_preds[: min(
                        n_frames, len(ids_for_best_preds))]
                else:
                    print("Uknown parameter")
                    sys.exit(-1)

                for id in ids_for_best_preds:
                    drive = driving_video[id]
                    pred = predictions[id]
                    triplet = np.stack([source_image[None, :, :, :],
                                        drive[None, :, :, :], pred[None, :, :, :]], axis=0)
                    name = '_'.join([str(n_source), str(n_video), str(id)])
                    for i, folder in enumerate(folders):
                        save(name, new_dataset_path, folder, triplet[i])
                    total += 1
                    if total >= N_total_images:
                        break


if __name__ == "__main__":
    in_args = parser.parse_args()
    path_to_fomm = in_args.fomm_path
    sys.path.append(path_to_fomm)

    if_clean_folders = in_args.if_clean_folders
    folders_to_store = in_args.folders_to_store
    new_dataset_path = in_args.new_dataset_path

    assert len(folders_to_store) == total_dirs

    create_folders(folders_to_store, new_dataset_path)
    if if_clean_folders:
        clean_folders(folders_to_store, new_dataset_path)

    config_path = in_args.config_path
    chekpoint_path = in_args.chekpoint_path
    generator, kp_detector = load_checkpoints(config_path=config_path,
                                              checkpoint_path=chekpoint_path)

    step_video = in_args.step_video
    N_total_images = in_args.N_total_images
    assert step_video > 0 and N_total_images > 0

    video_path = in_args.video_path
    img_path = in_args.img_path
    assert os.path.isdir(video_path) and os.path.isdir(img_path)

    face_comparison_mode = in_args.face_comparison_mode
    dataset_step_mode = in_args.dataset_step_mode
    n_frames = in_args.n_frames
    grid_step = in_args.grid_step

    assert grid_step > 0 and n_frames > 0
    assert face_comparison_mode == 'recognition' or face_comparison_mode == 'keypoints'
    assert dataset_step_mode == 'grid' or dataset_step_mode == 'most_distant'
    generate_data(generator, kp_detector, N_total_images, step_video,
                  video_path, img_path, new_dataset_path, folders_to_store,
                  face_comparison_mode, dataset_step_mode, n_frames, grid_step)
