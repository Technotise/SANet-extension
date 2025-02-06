import torch
import numpy as np
import pandas as pd
from net_p3d import P3DNetModified
from net_graph import GraphNet
from scipy.ndimage import zoom

def extend_dataframe_with_skeletons(df):
    """
    Extend each frame in the clips in the DataFrame with the skeleton map as the fourth channel.
    Args:
        df: pandas DataFrame with columns ['tensor', 'heatmap']
    Returns:
        DataFrame with the 'tensor' column updated to include skeleton channels (RGBS).
    """
    import numpy as np

    new_tensor_list = []

    for idx, row in df.iterrows():
        tensor = row['tensor']  # Shape (16, 200, 200, 3)
        heatmap = row['heatmap']  # Shape (16, 14, 50, 50)

        rgb_s_frames = []  # List to store the RGBS frames for this clip

        # Loop through each frame in the clip
        for frame_idx in range(tensor.shape[0]):
            frame_rgb = tensor[frame_idx]  # RGB frame, shape (200, 200, 3)
            joint_heatmap = heatmap[frame_idx]  # Joint heatmaps for the frame, shape (14, 50, 50)

            # Aggregate joint heatmaps to form a skeleton heatmap by averaging across joints
            skeleton_heatmap = np.mean(joint_heatmap, axis=0)  # Shape (50, 50)

            # Add skeleton as the fourth channel
            heatmap_resized = zoom(
                skeleton_heatmap,
                (frame_rgb.shape[0] / skeleton_heatmap.shape[0], 
                 frame_rgb.shape[1] / skeleton_heatmap.shape[1]),
                order=1
            )

            heatmap_channel = heatmap_resized[:, :, np.newaxis]  # (200, 200, 1)
            rgb_s_frame = np.concatenate((frame_rgb, heatmap_channel), axis=2)  # (200, 200, 4)
            rgb_s_frames.append(rgb_s_frame)

        # Stack all frames back into a single tensor of shape (16, 200, 200, 4)
        rgb_s_clip = np.stack(rgb_s_frames, axis=0)

        # Append to the new tensor list
        new_tensor_list.append(rgb_s_clip)

    # Update the DataFrame with the new tensors
    df['tensor'] = new_tensor_list
    return df

def get_outputs_cliprep(df: pd.DataFrame, factor: float):
    p3dnet_modified = P3DNetModified(factor=factor)
    outputs_list = []

    for idx, row in df.iterrows():
        try:
            rgb_s_clip = row['tensor']
            rgb_s_tensor = torch.tensor(rgb_s_clip, dtype=torch.float).permute(3, 0, 1, 2)
            rgb_s_tensor = rgb_s_tensor.unsqueeze(0)
            outputs = p3dnet_modified(rgb_s_tensor)
            outputs_list.append(outputs.cpu().detach().numpy())
        except Exception as e:
            print(f"Error in row {idx}: {e}")
            continue

    df['outputs_cliprep'] = outputs_list
    return df

def get_outputs_clipscl(df: pd.DataFrame, device: torch.device):
    graphnet_model = GraphNet(in_channels=1, device=device)
    graphnet_model.eval()

    outputs_list = []
    attention_list = []

    for idx, row in df.iterrows():
        heatmap_clip = row['heatmap']
        heatmap_tensor = torch.tensor(heatmap_clip, dtype=torch.float)
        pooled_heatmap = torch.nn.functional.adaptive_avg_pool2d(heatmap_tensor, (1, 1))
        graph_input = pooled_heatmap.squeeze(-1).squeeze(-1).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs, attention = graphnet_model(graph_input)

        outputs_list.append(outputs.cpu().detach().numpy())
        attention_list.append(attention.cpu().detach().numpy())

    df['outputs_clipscl'] = outputs_list
    df['clipscl_attention_weights'] = attention_list
    return df

def run_combined_model(dataframe, model):
    videos = dataframe['video'].tolist()
    frames = torch.tensor(np.stack(dataframe['outputs_frmske'].values)).float()
    rgb_skeleton_concat = torch.tensor(np.stack(dataframe['outputs_cliprep'].values)).float()
    skeleton_data = torch.tensor(np.stack(dataframe['outputs_clipscl'].values)).float()

    frames = torch.mean(frames, dim=2).squeeze(1)
    device = next(model.parameters()).device
    frames = frames.to(device)
    rgb_skeleton_concat = rgb_skeleton_concat.to(device)
    skeleton_data = skeleton_data.to(device)

    with torch.no_grad():
        output = model(frames, rgb_skeleton_concat, skeleton_data)

    return videos, output
