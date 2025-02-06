import os
import glob
import pandas as pd
from torch import nn as nn
from torch.nn import functional as F

class Vgg(nn.Module):
    def __init__(self, factor, num_joints=14, n_classes=10):
        super(Vgg, self).__init__()
        self.factor = factor
        self.conv1_1 = nn.Conv2d(3, int(64*self.factor), kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(int(64*self.factor), int(64*self.factor), kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(int(64*self.factor), int(128*self.factor), kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(int(128*self.factor), int(128*self.factor), kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(int(128*self.factor), int(256*self.factor), kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(int(256*self.factor), int(256*self.factor), kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(int(256*self.factor), int(256*self.factor), kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(int(256*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(int(512*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(int(512*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(int(512*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(int(512*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(int(512*self.factor), int(512*self.factor), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc6 = nn.Linear(6*6*int(512*factor), n_classes)
        self.Tranconv1 = nn.ConvTranspose2d(int(512*self.factor), int(256*self.factor), 3, stride=2)
        self.Tranconv2 = nn.ConvTranspose2d(int(256*self.factor), int(128*self.factor), 3, stride=2, padding=1, output_padding=1)
        self.pointwise1 = nn.Conv2d(int(128*factor), num_joints, 1)
        self.Tranconv3 = nn.ConvTranspose2d(int(256 * self.factor), int(128 * self.factor), 3, stride=2, padding=1, output_padding=1)
        self.pointwise2 = nn.Conv2d(int(128*self.factor), num_joints, 1)

    def Conv1(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        return x

    def Conv2(self, x):
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        return x

    def Conv3(self, x):
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        return x

    def Conv4(self, x):
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        return x

    def Conv5(self, x):
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch, video_length, w, h, channel = x.shape
        x = x.reshape(-1, w, h, channel).permute(0, 3, 1, 2).contiguous()
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        heatmap1 = self.Tranconv3(x)
        heatmap1 = self.pointwise2(heatmap1)
        x = self.Conv4(x)
        heatmap = self.Tranconv1(x)
        heatmap = self.Tranconv2(heatmap)
        heatmap = self.pointwise1(heatmap)
        heatmap = heatmap1 + heatmap
        x = self.Conv5(x)
        x = x.view(-1, 6 * 6 * int(512*self.factor))
        x = self.fc6(x)
        x = x.view(batch, video_length, -1)
        return x, heatmap

def vggnet_bridge(preprocessed_data_folder, factor, device, result_save_folder):
    pickle_folder = os.path.join(preprocessed_data_folder, "pickle")
    if not os.path.exists(pickle_folder):
        raise FileNotFoundError(f"Pickle folder not found: {pickle_folder}")
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)
    model = Vgg(factor, n_classes=2048).to(device)
    pickle_files = glob.glob(f"{pickle_folder}/*.pkl")
    for idx, file in enumerate(pickle_files):
        df_batch = pd.read_pickle(file)
        outputs_list = []
        heatmap_list = []
        for _, row in df_batch.iterrows():
            clip_tensor = row['tensor'].to(device).unsqueeze(0)
            outputs, heatmap = model(clip_tensor)
            outputs_list.append(outputs.cpu().detach().numpy())
            heatmap_list.append(heatmap.cpu().detach().numpy())
        df_batch['outputs_frmske'] = outputs_list
        df_batch['heatmap'] = heatmap_list
        result_file = os.path.join(result_save_folder, f"result_{os.path.basename(file)}")
        df_batch.to_pickle(result_file)
