#!/usr/bin/env python

import argparse
import pathlib

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import tqdm
from oct2py import octave as oct

screen_size = [[1280,800],[1440,900],[1280,800],[1440,900],[1280,800],[1440,900],[1680,1050],[1440,900],[1440,900],[1440,900],[1440,900],[1280,800],[1280,800],[1280,800],[1440,900]]

def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw]).astype(np.float32)


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]).astype(np.float32)


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path,
                     delimiter=' ',
                     header=None,
                     names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def save_one_person(person_id: str, data_dir: pathlib.Path,
                    eval_dir: pathlib.Path, output_path: pathlib.Path,i: int) -> None:
    left_images = dict()
    left_poses = dict()
    left_gazes = dict()
    right_images = dict()
    right_poses = dict()
    right_gazes = dict()
    right_target = dict() #Added
    left_target = dict()  #Added
    filenames = dict()
    person_dir = data_dir / person_id

    #print("Working with this dataset")
    #print(person_dir)
    #print("Working with this person")
    #print(person_id)

    for path in sorted(person_dir.glob('*')):
        #print("Working with this file: ",path.as_posix())
        mat_data = scipy.io.loadmat(path.as_posix(),
                                    struct_as_record=False,
                                    squeeze_me=True)
        data = mat_data['data']

        day = path.stem
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze


        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze


        right_target[day] = data.right.target /screen_size[i] #Added
        left_target[day] = data.left.target/ screen_size[i]  # Added

        #print("Right target:",right_target[day])

        filenames[day] = mat_data['filenames']

        if not isinstance(filenames[day], np.ndarray):
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            left_target[day] = np.array(left_target[day])       #Added
            right_target[day] = np.array(right_target[day])     #Added
            filenames[day] = np.array([filenames[day]])

    df = get_eval_info(person_id, eval_dir)
    images = []
    poses = []
    gazes = []
    targets = []  #Added

    for _, row in df.iterrows():
        day = row.day
        #print("Day==>",day)
        #print(filenames[day])
        #print(row.filename)

        index = np.where(filenames[day] == row.filename)[0][0]
        if row.side == 'left':
            image = left_images[day][index]

            pose = convert_pose(left_poses[day][index])

            gaze = convert_gaze(left_gazes[day][index])

            target = left_target[day][index]

        else:
            image = right_images[day][index][:, ::-1]

            pose = convert_pose(right_poses[day][index]) * np.array([1, -1])

            gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])

            target = right_target[day][index]

        images.append(image)
        poses.append(pose)
        #gazes.append(gaze)
        targets.append(target)                                 #Added

    images = np.asarray(images).astype(np.uint8)
    poses = np.asarray(poses).astype(np.float32)
    #gazes = np.asarray(gazes).astype(np.float32)
    targets = np.asarray(targets).astype(np.float32)              #Added

    with h5py.File(output_path, 'a') as f_output:
        f_output.create_dataset(f'{person_id}/image', data=images)
        f_output.create_dataset(f'{person_id}/pose', data=poses)
        #f_output.create_dataset(f'{person_id}/gaze', data=gazes)
        f_output.create_dataset(f'{person_id}/target', data=targets)        #Added

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'MPIIGaze.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)
    i = 0
    for person_id in tqdm.tqdm(range(15)):                  #for one person out of 15 persons
        person_id = f'p{person_id:02}'
        data_dir = dataset_dir / 'Data' / 'Normalized'      # Taking the Normalized data in the MPIIGaze dataset
        eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'  # For each person a soup is being created by randomly picking 3000 images
        save_one_person(person_id, data_dir, eval_dir, output_path,i)
        i = i+1

if __name__ == '__main__':
    main()
