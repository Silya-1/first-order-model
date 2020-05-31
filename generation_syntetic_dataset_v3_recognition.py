
import sys
import os
import glob
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.autograd import Variable

from skimage import img_as_ubyte
from skimage.transform import resize
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from PIL import Image
import imageio

import face_recognition
import face_alignment

from scipy.spatial import ConvexHull
from scipy import linalg

from demo import load_checkpoints, make_animation, find_best_frame

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Process neccesary args')

    parser.add_argument('--img_path', type=str, required=True, help='Path to image dir')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to new dataset folder')
    parser.add_argument('--chekpoint_path', default='./content/vox-cpk.pth.tar',
                        help='Path to checkpoint')
    parser.add_argument('--config_path', type=str, default='./config/vox-256.yaml',
                        help='Path to config folder')
    
    parser.add_argument('--step_video', type=int, default=30,
                        help='Step in video grid')
    parser.add_argument('--dataset_step_mode', type=str, default='grid',
                        help='Mode to choose frames', choices=['grid', 'most_distant'])
    parser.add_argument('--grid_step', type=int, default=2,
                        help='Grid step')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='Number of most distant frames')
    parser.add_argument('--face_comparison_mode', type=str, default='recognition',
                        help='Mode to select distance', choices=['recognition', 'keypoints'])
    parser.add_argument('--N_total_images', type=int, default=10,
                        help='Number of total images to generate')

    parser.add_argument('--folders_to_store', default=['source', 'drive', 'predict'],
                        help='Name of folders for new dataset')
    parser.add_argument('--no_clean_folders', action='store_false', dest='if_clean_folders',
                        help='Flag to clear folders')
    parser.set_defaults(if_clean_folders=True)
    
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    return args


def check_args(args):
    """ Check the args for valid"""
    assert len(args.folders_to_store) == 3, 'Check the option `folders_to_store`' 
    assert args.step_video > 0, 'Number is not valid, must be grather then 0'
    assert args.N_total_images > 0, 'Number is not valid, must be grather then 0'
    assert os.path.isdir(args.video_path), 'Path for directory with videos is not valid'
    assert os.path.isdir(args.img_path), 'Path for directory with images is not valid'
    assert args.grid_step > 0, 'Number is not valid, must be grather then 0'
    assert args.n_frames > 0, 'Number is not valid, must be grather then 0'
    
    args.N_video = len(os.listdir(args.video_path))
    args.N_img = len(os.listdir(args.img_path))
    args.N_img_by_video = args.N_total_images // (args.N_video * args.n_frames) + 1
    return args
    

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


def path_getter(x):
    return list(map(lambda y: os.path.join(os.path.abspath(x), y), os.listdir(x)))


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


def generate_data(generator, kp_detector, N_total_images, N_img_by_video, step_video,
                  video_path, img_path, new_dataset_path, folders, face_comparison_mode,
                  dataset_step_mode, n_frames, grid_step):
    total = 0
    N_image_by_video = N_total_images // ( N )
    with tqdm(enumerate(path_getter(video_path))) as tq:
        for n_video, video_path in tq:
            if total >= N_total_images:
                break
            driving_video = imageio.mimread(video_path, memtest=False)
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

            ids = np.arange(0, len(driving_video), step_video)
            driving_video = np.array(driving_video)[ids]
            
            random_img = np.random.choice(path_getter(img_path), size=N_img_by_video, replace=False)
            for n_source, source_path in enumerate(random_img):
                if total >= N_total_images:
                    break
                source_image = imageio.imread(source_path)
                source_image = resize(source_image, (256, 256))[..., :3]

                try:
                    norms_for_best_source = get_key_points(source_image, driving_video)
                except Exception:
                    print(f' Image not processed by get_key_points: {img_path} - {video_path}')
                    continue

                best_i = np.argmin(norms_for_best_source)
                driving_video[0], driving_video[best_i] = driving_video[best_i], driving_video[0]

                predictions = make_animation(source_image, driving_video,
                                             generator, kp_detector, relative=True,
                                             adapt_movement_scale=True)

                if face_comparison_mode == 'keypoints':
                    try:
                        norms_for_best_preds = get_key_points(source_image, predictions)
                    except Exception:
                        print(f' Image not processed by get_key_points: {img_path} - {video_path}')
                        continue

                elif face_comparison_mode == 'recognition':
                    try:
                        norms_for_best_preds = recognition(
                            source_image, predictions)
                    except Exception:
                        print(f' Image not processed by recognition_net: {img_path} - {video_path}')
                        continue
                else:
                    print("Uknown parameter")
                    sys.exit(-1)

                ids_for_best_preds = np.argsort(norms_for_best_preds)[::-1]

                if dataset_step_mode == 'grid':
                    ids_for_best_preds = ids_for_best_preds[: len(ids_for_best_preds): grid_step]
                elif dataset_step_mode == 'most_distant':
                    ids_for_best_preds = ids_for_best_preds[: min(n_frames, len(ids_for_best_preds))]
                else:
                    print("Uknown parameter")
                    sys.exit(-1)

                for id in ids_for_best_preds:
                    drive = driving_video[id]
                    pred = predictions[id]
                    triplet = np.stack([source_image[None, :, :, :],
                                        drive[None, :, :, :],
                                        pred[None, :, :, :]], axis=0)
                    name = '_'.join([str(n_source), str(n_video), str(id)])
                    for i, folder in enumerate(folders):
                        save(name, new_dataset_path, folder, triplet[i])
                    total += 1
                    if total >= N_total_images:
                        break
                tq.set_postfix({'total_img': f'{total}'})
                
            tq.set_postfix({'total_img': f'{total}'})
                            
def main():
    args = parse_args()
    args = check_args(args)
    
    print(f'Raw Images total: {args.N_img}')
    print(f'Raw Videos total: {args.N_video}')
    print(f'Ready to generate {args.N_total_images} image')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    create_folders(args.folders_to_store, args.output_path)
    if args.if_clean_folders:
        clean_folders(args.folders_to_store, args.output_path)
    
    generator, kp_detector = load_checkpoints(config_path=args.config_path,
                                              checkpoint_path=args.chekpoint_path)

    generate_data(generator=generator,
                  kp_detector=kp_detector,
                  N_total_images=args.N_total_images,
                  N_img_by_video=args.N_img_by_video,
                  step_video=args.step_video,
                  video_path=args.video_path,
                  img_path=args.img_path,
                  new_dataset_path=args.output_path,
                  folders=args.folders_to_store,
                  face_comparison_mode=args.face_comparison_mode,
                  dataset_step_mode=args.dataset_step_mode,
                  n_frames=args.n_frames,
                  grid_step=args.grid_step)



if __name__ == "__main__":
    print('-- start --')
    main()
    print('--  end  --')
