import os
import pandas as pd

import torch
from torch.utils import data


ACTIVITY_DESCRIPTIONS = {
    "STD": "Standing",
    "WAL": "Walking",
    "JOG": "Jogging",
    "JUM": "Jumping",
    "STU": "Stairs up",
    "STN": "Stairs down",
    "SCH": "Stand to sit",
    "SIT": "Sitting",
    "CHU": "Sit to stand",
    "CSI": "Car step in",
    "CSO": "Car step out",
}


SCENARIO_DESCRIPTIONS = {
    "SLH": "Leaving home",
    "SBW": "Being at work",
    "SLW": "Leaving work",
    "SBE": "Exercising",
    "SRH": "Returning home",
}


LABEL_CODES = {
    "STD": 0,
    "WAL": 1,
    "JOG": 2,
    "JUM": 3,
    "STU": 4,
    "STN": 5,
    "SCH": 6,
    "SIT": 7,
    "CHU": 8,
    "CSI": 9,
    "CSO": 10,
}


class MobiActV2(data.Dataset):
    """ A pyTorch Dataset class for MobiActV2 frames.
    
    Args:
        root (string): The root directory where the dataset exists.
        sensors (string): The sensor channels to include.
        users (list): List of user IDs to include in the dataset instance.
    """

    def __init__(self, root, sensors, users):
        self.root = root
        self.sensors = sensors
        self.users = users
        self.activity_descriptions = ACTIVITY_DESCRIPTIONS
        self.scenario_descriptions = SCENARIO_DESCRIPTIONS
        self.label_codes = LABEL_CODES
        self._cache = {}

        # Load files for given users and activities/scenarios of interest
        files = os.listdir(self.root)
        files = [f for f in files if int(f.split("_")[1]) in self.users]
        files = [
            f
            for f in files
            if f.split("_")[0] in ACTIVITY_DESCRIPTIONS
            or f.split("_")[0] in SCENARIO_DESCRIPTIONS
        ]
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Check if frame is in cache
        if idx in self._cache:
            return self._cache[idx][0], self._cache[idx][1]

        # Load frame data
        file_path = os.path.join(self.root, self.files[idx])
        df = pd.read_csv(file_path)

        # Normalize sensor data to Android sensor ranges
        # Acceleration: [-20, 20] (m/s**2)
        # Angular velocity: [-10, 10] (rad/s)
        # Rotation:
        #     - Azimuth: [0, 360] (degrees)
        #     - Pitch: [-180, 180] (degrees)
        #     - Roll: [-90, 90] (degrees)
        df[["acc_x", "acc_y", "acc_z"]] = df[["acc_x", "acc_y", "acc_z"]].apply(
            lambda x: -1 + 2 * (x + 20) / 40
        )
        df[["gyro_x", "gyro_y", "gyro_z"]] = df[["gyro_x", "gyro_y", "gyro_z"]].apply(
            lambda x: -1 + 2 * (x + 10) / 20
        )
        df["azimuth"] = df["azimuth"].apply(lambda x: -1 + 2 * x / 360)
        df["pitch"] = df["pitch"].apply(lambda x: -1 + 2 * (x + 180) / 360)
        df["roll"] = df["roll"].apply(lambda x: -1 + 2 * (x + 90) / 180)

        # Slice X, y
        X = df[self.sensors]
        y = df["label"]
        ts = df["rel_time"].tolist()

        # Tensorize
        X = torch.tensor(X.values, dtype=torch.float64)
        file_id = torch.tensor([idx])
        y = torch.tensor(list(map(lambda x: self.label_codes[x], y)), dtype=torch.int64)

        target = {}
        target["file_id"] = file_id
        target["filename"] = self.files[idx]
        target["y"] = y
        target["ts"] = ts

        # Add frame to cache
        if idx not in self._cache:
            self._cache[idx] = [X, target]

        return X, target
