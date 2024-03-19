import os
import numpy as np
import cv2
import h5py
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from constants import DT
from pathlib import Path
import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

import os
import glob
import zipfile

def zip_mp4_files(directory):
    # Path to the zip file we want to create, under the input directory
    zip_path = os.path.join(directory, 'all_videos.zip')
    
    # Create a zip file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Change the current directory to the input directory to ensure file names
        # in the zip archive are relative to the input directory
        os.chdir(directory)
        
        # Loop through all .mp4 files in the directory
        for file in glob.glob('*.mp4'):
            # Add file to the zip file
            zipf.write(file)
    
    print(f'All .mp4 files have been zipped into {zip_path}')
    # delete all .mp4 files
    directory = Path(directory)
    for file in directory.glob('*.mp4'):
        os.remove(file)


def load_hdf5(dataset_path):
    # dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        dataset_path = dataset_path + '.hdf5'
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def save_hdf5_video(path):
    qpos, qvel, action, image_dict = load_hdf5(path)
    save_videos(image_dict, DT, video_path=os.path.join(path.replace('.hdf5','') + '.mp4'))

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    ismirror = args['ismirror']
    if ismirror:
        data_name = f'mirror_episode_{episode_idx}'
    else:
        data_name = f'episode_{episode_idx}'

    if episode_idx is not None:
        save_hdf5_video(os.path.join(dataset_dir, data_name))
    else:
        print(f'Visualizing all episodes in {dataset_dir}')
        data_names = [name for name in os.listdir(dataset_dir) if name.endswith('.hdf5')]
        with multiprocessing.Pool(16) as pool:
            pool.map(save_hdf5_video, [os.path.join(dataset_dir, name) for name in data_names])
            
    zip_mp4_files(dataset_dir)
    # qpos, qvel, action, image_dict = load_hdf5(dataset_dir, datas_name)
    # save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, datas_name + '_video.mp4'))
    # visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))
