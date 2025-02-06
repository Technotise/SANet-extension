import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

# Define skeleton connections between keypoints for the 14-point skeleton
# SKELETON_CONNECTIONS=[
#    ("left_shoulder", "right_shoulder"),  # Neck (approximated as the midpoint between shoulders)
#    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
#    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
#    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
#    ("nose", "left_eye"), ("nose", "right_eye"),
#    ("left_eye", "left_ear"), ("right_eye", "right_ear")
#]


# Initialize MediaPipe pose detection
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(static_image_mode=True)

# Define keypoints and their indices in MediaPipe for the desired 14 keypoints
KEYPOINT_INDICES={
    "nose": 0,
    "left_eye": 3,
    "right_eye": 6,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

def preprocess(step, input_folder, preprocessed_data_folder, target_count, image_size=(200, 200), window_size=16, stride=8):
    if step <= 0:
        # Step 0: Copy the original folder to a new folder for preprocessing
        print(f"Start: Copy folder")
        if os.path.exists(preprocessed_data_folder):
            shutil.rmtree(preprocessed_data_folder)
        shutil.copytree(input_folder, preprocessed_data_folder)
        print(f"End: Copy folder")

    if step <= 1:
        # Step 1: Resize PNG images
        print(f"Start: Resizing")
        for root, _, files in os.walk(preprocessed_data_folder):

            if 'skip_string' in root:
                continue
            for file in files:
                if file.lower().endswith(".png"):
                    # Get the full path of the image
                    input_image_path=os.path.join(root, file)
                    # Load and resize the image
                    try:
                        with Image.open(input_image_path) as img:
                            img_resized=img.resize(image_size, Image.LANCZOS)
                            img_resized.save(input_image_path)
                            # print(f"Resized and saved: {input_image_path}")
                    except Exception as e:
                        print(f"Error processing {input_image_path}: {e}")
        print(f"End: Resizing")

    if step <= 2:
        # Step 2: Add Gaussian noise to resized images
        print(f"Start: Gaussian noise")
        for root, _, files in os.walk(preprocessed_data_folder):
            if 'skip_string' in root:
                continue
            for file in files:
                if file.lower().endswith(".png"):
                    # Get the full path of the image
                    input_image_path=os.path.join(root, file)
                    # Load the image and add Gaussian noise
                    try:
                        with Image.open(input_image_path) as img:
                            # img = img.convert('L')  # Convert to grayscale if needed
                            img_array=np.array(img)

                            # Generate Gaussian noise
                            mean=0
                            std_dev=12
                            noise=np.random.normal(mean, std_dev, img_array.shape)
                            noised_img_array=img_array + noise

                            # Clip the values to be in valid range
                            noised_img_array=np.clip(noised_img_array, 0, 255).astype(np.uint8)

                            # Convert back to image
                            noised_img=Image.fromarray(noised_img_array)
                            noised_img.save(input_image_path)
                            # print(f"Added Gaussian noise and saved: {input_image_path}")
                    except Exception as e:
                        print(f"Error processing {input_image_path}: {e}")
        print(f"End: Gaussian noise")

    if step <= 3:
        # Step 3: Ensure each subfolder has a constant number of images
        print(f"Start: Ensure each subfolder has a constant number of images")
        for root, _, files in os.walk(preprocessed_data_folder):
            if 'skip_string' in root:
                continue
            png_files=sorted([f for f in files if f.lower().endswith(".png")])
            num_files=len(png_files)

            # If there are fewer than target_count images, add black images
            if num_files > 2:
                if num_files < target_count:
                    last_index=num_files + 1  # int(png_files[-1].split("images")[-1].split(".png")[0]) if num_files > 0 else 0
                    for i in range(num_files, target_count):
                        black_img=Image.new('RGB', image_size, (0, 0, 0))
                        new_file_name=f"images{i + 1:04d}.png"
                        black_image_path=os.path.join(root, new_file_name)
                        black_img.save(black_image_path)
                        # print(f"Added black image: {black_image_path}")

                # If there are more than target_count images, delete the extra images
                elif num_files > target_count:
                    for file in png_files[target_count:]:
                        os.remove(os.path.join(root, file))
                        # print(f"Deleted extra image: {os.path.join(root, file)}")
        print(f"End: Ensure each subfolder has a constant number of images")

    if step <= 4:
        # Step 4: Generate and cache MediaPipe heatmaps for each frame
        for root, _, files in os.walk(preprocessed_data_folder):
            if 'skip_string' in root:
                continue

        # Create the main `mediapipe_heatmaps` folder at the root of `output_base_folder`
        print(f"Start: Create heatmaps")
        main_heatmap_folder=os.path.join(preprocessed_data_folder, "mediapipe_heatmaps")
        os.makedirs(main_heatmap_folder, exist_ok=True)

        for root, _, files in os.walk(preprocessed_data_folder):
            if 'skip_string' in root:
                continue

            # Skip existing heatmap folders
            if root == main_heatmap_folder or main_heatmap_folder in root:
                continue

            video_folder_name=os.path.basename(root)
            heatmap_folder=os.path.join(main_heatmap_folder, video_folder_name)

            # Check for a marker file to skip the folder
            marker_file=os.path.join(heatmap_folder, "processed.marker")
            if os.path.exists(marker_file):
                # print(f"Skipped folder (already processed): {heatmap_folder}")
                continue

            os.makedirs(heatmap_folder, exist_ok=True)

            for file in sorted(files):
                if file.lower().endswith(".png"):
                    frame_path = os.path.join(root, file)
                    frame=cv2.imread(frame_path)
                    frame_heatmaps=generate_keypoints_heatmaps(frame, image_size=image_size, heatmap_size=(50, 50),
                                                               sigma=5)

                    heatmap_filename=os.path.splitext(file)[0] + ".npy"
                    heatmap_path=os.path.join(heatmap_folder, heatmap_filename)
                    np.save(heatmap_path, frame_heatmaps)

            # Create marker file after processing the folder
            with open(marker_file, "w") as f:
                f.write("processed")
            print(f"Finished processing folder: {heatmap_folder}")

        print(f"End: Create heatmaps")

    if step <= 5:
        # Step 5: Create sliding window clips
        print(f"Start: Create sliding window clips")
        for root, dirs, files in os.walk(preprocessed_data_folder):
            # Verarbeite nur eine Ebene unterhalb von output_base_folder
            if root.count(os.sep) != preprocessed_data_folder.count(os.sep) + 1:
                continue

            if 'skip_string' in root:
                continue

            png_files=sorted([f for f in files if f.lower().endswith(".png")])
            num_files=len(png_files)

            # Überprüfen, ob es bereits den maximalen Clip-Ordner gibt
            max_clip_folder=os.path.join(root, f"clip_{(num_files - window_size) // stride + 1:02d}")
            if os.path.exists(max_clip_folder):
                print(f"Skipping {root}: All clips already exist.")
                continue

            print(f"Processing folder: {root}")  # Nur Mainfolder loggen

            clip_num=1
            for start in range(0, num_files - window_size + 1, stride):
                clip_folder=os.path.join(root, f"clip_{clip_num:02d}")
                if not os.path.exists(clip_folder):
                    os.makedirs(clip_folder)

                for i in range(start, start + window_size):
                    input_image_path=os.path.join(root, png_files[i])
                    output_image_path=os.path.join(clip_folder, png_files[i])
                    with Image.open(input_image_path) as img:
                        img.save(output_image_path)

                clip_num+=1
        print(f"End: Create sliding window clips")

    if step <= 6:
        pickle_folder = os.path.join(preprocessed_data_folder, "pickle")  # Ändere den Pfad

        if not os.path.exists(pickle_folder):
            os.makedirs(pickle_folder)

        load_clips(preprocessed_data_folder, pickle_folder)


def generate_keypoints_heatmaps(frame, image_size=(200, 200), heatmap_size=(50, 50), sigma=5):
    frame=cv2.resize(frame, image_size)
    height, width=heatmap_size
    results=pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    num_keypoints=len(KEYPOINT_INDICES) + 1  # +1 for the neck point
    heatmaps=np.zeros((num_keypoints, height, width), dtype=np.float32)

    if results.pose_landmarks:
        landmarks=results.pose_landmarks.landmark

        # Create heatmaps for each specific keypoint
        for idx, key in enumerate(KEYPOINT_INDICES.keys()):
            keypoint_index=KEYPOINT_INDICES[key]
            landmark=landmarks[keypoint_index]
            x=int(landmark.x * width)
            y=int(landmark.y * height)
            if 0 <= x < width and 0 <= y < height:
                heatmap=np.zeros((height, width), dtype=np.float32)
                heatmap[y, x]=1
                heatmaps[idx]=gaussian_filter(heatmap, sigma=sigma)

        # Calculate neck position as the midpoint between left and right shoulders
        left_shoulder=landmarks[KEYPOINT_INDICES["left_shoulder"]]
        right_shoulder=landmarks[KEYPOINT_INDICES["right_shoulder"]]
        neck_x=int((left_shoulder.x + right_shoulder.x) / 2 * width)
        neck_y=int((left_shoulder.y + right_shoulder.y) / 2 * height)
        if 0 <= neck_x < width and 0 <= neck_y < height:
            heatmap=np.zeros((height, width), dtype=np.float32)
            heatmap[neck_y, neck_x]=1
            heatmaps[-1]=gaussian_filter(heatmap, sigma=sigma)  # Last index is for the neck

    return heatmaps

def load_clips(base_folder, save_path):
    print("Def: load_clips")
    total_clips = 0 
    total_videos = 0
    batch_size = 500
    batch_data = []

    for video_idx, video_folder in enumerate(os.listdir(base_folder)):
        video_path = os.path.join(base_folder, video_folder)

        if os.path.isdir(video_path) and video_folder != "mediapipe_heatmaps":
            total_videos += 1
            print(f"Processing video folder {total_videos}: {video_folder}")

            for clip_idx, clip_folder in enumerate(os.listdir(video_path)):
                clip_path = os.path.join(video_path, clip_folder)

                if os.path.isdir(clip_path) and clip_folder.startswith("clip_"):
                    clip_images = []
                    for image_file in sorted(os.listdir(clip_path)):
                        if image_file.lower().endswith(".png"):
                            image_path = os.path.join(clip_path, image_file)
                            with Image.open(image_path) as img:
                                img = img.convert('RGB')
                                img_array = np.array(img).astype(
                                    np.float32) / 255.0
                                clip_images.append(img_array)

                    if len(clip_images) == 16:
                        clip_images_np = np.array(clip_images)
                        clip_tensor = torch.tensor(clip_images_np)
                        batch_data.append({
                            "video": video_folder,
                            "clip": clip_folder,
                            "tensor": clip_tensor
                        })
                        total_clips += 1

                        if len(batch_data) >= batch_size:
                            print(f"Saving batch of {len(batch_data)} clips...")
                            batch_df = pd.DataFrame(batch_data)
                            batch_df.to_pickle(f"{save_path}/clips_batch_{total_clips // batch_size}.pkl")
                            batch_data = []

    if batch_data:
        print(f"Saving final batch of {len(batch_data)} clips...")
        batch_df = pd.DataFrame(batch_data)
        batch_df.to_pickle(f"{save_path}/clips_batch_final.pkl")

    print(f"Total video folders processed: {total_videos}")
    print(f"Total clips processed: {total_clips}")
