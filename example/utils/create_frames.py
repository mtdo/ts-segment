import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

SOURCE_DIR = 'data/MobiActV2/sessions/csv'
DESTINATION_DIR = 'data/MobiActV2/frames'
INTERPOLATION_FREQUENCY = 50 # Hz
FRAME_LENGTH = 3 # seconds
N_POINTS_FRAME = int(INTERPOLATION_FREQUENCY*FRAME_LENGTH)


if not os.path.exists(DESTINATION_DIR):
    print(f"Creating destination dir: {DESTINATION_DIR}")
    os.mkdir(DESTINATION_DIR)
    
print(f"Creating frames from dir: {SOURCE_DIR}")
files = np.array(os.listdir(SOURCE_DIR))
files = [f for f in files if f.endswith('.csv')]
for file in tqdm(files):
    filepath = os.path.join(SOURCE_DIR, file)
    df = pd.read_csv(filepath)
    
    # Simple data validity checks
    ##### Check if relative time values are valid
    if df.rel_time.min() != 0 and df.rel_time.max() > df.rel_time.min():
        print(f"Found file with a non valid minimum 'rel_time' start: {file}")
    ##### Check for missing sensor channels
    if not set(['acc_x', 'acc_y', 'acc_z']).issubset(df.columns):
        print(f"Found file with incomplete accelerometer channels: {file}")
        print(f"Omitting...")
        continue
    if not set(['gyro_x', 'gyro_y', 'gyro_z']).issubset(df.columns):
        print(f"Found file with incomplete gyroscope channels: {file}")
        print(f"Omitting...")
        continue
    if not set(['azimuth', 'pitch', 'roll']).issubset(df.columns):
        print(f"Found file with incomplete orientation channels: {file}")
        print(f"Omitting...")
        continue
        
    # There is a bug in the data where the azimuth angle is sometimes negative (<<1% of the data)
    # This can be easily fixed by taking the modulo of 360
    df.azimuth = df.azimuth%360
        
    # Interpolate data to specified interpolation frequency
    ##### Labels and numerical data
    labels = df[['rel_time', 'label']]
    df = df.drop(['timestamp', 'label'], axis=1)
    
    ##### Interpolation of numerical values
    n_points = (df.rel_time.max() - df.rel_time.min())*INTERPOLATION_FREQUENCY
    interp_f = interp1d(df.index, df.values, kind='slinear', assume_sorted=True, axis=0)
    x_prime = np.linspace(df.index.min(), df.index.max(), n_points)
    y_prime = interp_f(x_prime).astype(np.float32)
    df = pd.DataFrame(y_prime, columns=df.columns)
    
    ##### Readdition of labels to df
    transitions = []
    transitions.append([labels.rel_time[0], labels.label[0]])
    for i in range(1, len(labels)):
        if labels.label[i] != labels.label[i-1]:
            transitions.append([labels.rel_time[i], labels.label[i]])
    transitions = [[np.where(df.rel_time >= x[0])[0][0], x[1]] for x in transitions]
    
    df['label'] = np.nan
    for transition in transitions:
        df.label.iloc[transition[0]] = transition[1]
    df = df.fillna(method='ffill')
    
    # Split interpolated df to 'frames' of specified length
    for frame_idx, i in enumerate(range(0, len(df)-N_POINTS_FRAME, N_POINTS_FRAME)):
        sub_df = df.iloc[i:i+N_POINTS_FRAME]
        fn = f"{os.path.join(DESTINATION_DIR, file[:-4])}_{frame_idx}.csv"
        assert len(sub_df) == N_POINTS_FRAME
        sub_df.to_csv(fn)
    sub_df = df.iloc[(len(df)-N_POINTS_FRAME):len(df)]
    fn = f"{os.path.join(DESTINATION_DIR, file[:-4])}_end.csv"
    assert len(sub_df) == N_POINTS_FRAME, str(len(df))
    sub_df.to_csv(fn)

print("Finished creating frames.")
print(f"Frames saved to dir: {DESTINATION_DIR}")
