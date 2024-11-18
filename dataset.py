import torch
from torch.utils.data import Dataset
from PIL import Image

# Example Dataset class that provides (photo, xray, label) pairs
class DentalDataset(Dataset):
    def __init__(self, photo_paths_L, photo_paths_U, xray_paths, labels, photo_transform=None, xray_transform=None):
        self.photo_paths_L = photo_paths_L
        self.photo_paths_U = photo_paths_U
        self.xray_paths = xray_paths
        self.labels = labels
        self.photo_transform = photo_transform
        self.xray_transform = xray_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        photo_L_path = self.photo_paths_L[idx]
        photo_U_path = self.photo_paths_U[idx]
        xray_path = self.xray_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 이미지를 해당 인덱스에서 로드합니다
        photo_L = self.load_image(photo_L_path, channels=3)
        photo_U = self.load_image(photo_U_path, channels=3)
        xray = self.load_image(xray_path, channels=1)

        # 필요하면 변환을 적용합니다
        if self.photo_transform:
            photo_L = self.photo_transform(photo_L)
            photo_U = self.photo_transform(photo_U)
        if self.xray_transform:
            xray = self.xray_transform(xray)

        return photo_L, photo_U, xray, label

    def load_image(self, path, channels):
        image = Image.open(path).convert('RGB' if channels == 3 else 'L')
        return image