import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load the raw data
columns = ['id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
train = pd.read_csv('train_FD001.txt', sep='\s+', names=columns)
test = pd.read_csv('test_FD001.txt', sep='\s+', names=columns)
true_rul = pd.read_csv('RUL_FD001.txt', sep='\s+', names=['RUL'])

# 2. Drop the flatline sensors and static settings
drop_cols = ['op3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

# 3. Calculate Piecewise RUL for training data (Cap at 125)
MAX_RUL = 125
train_max_cycle = train.groupby('id')['cycle'].max().reset_index()
train_max_cycle.columns = ['id', 'max_cycle']
train = train.merge(train_max_cycle, on=['id'], how='left')
train['RUL'] = train['max_cycle'] - train['cycle']
train['RUL'] = train['RUL'].clip(upper=MAX_RUL)
train.drop(columns=['max_cycle'], inplace=True)

# 4. Normalize the sensor data (Fit ONLY on train to avoid leakage)
features = train.columns.drop(['id', 'cycle', 'RUL'])
scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# 5. Build the Sliding Window function for CNN/LSTM
def generate_sequences(df, seq_length, feature_cols):
    seqs, labels = [], []
    for engine_id, group in df.groupby('id'):
        data = group[feature_cols].values
        if 'RUL' in group.columns:
            rul = group['RUL'].values
        # Only create sequences if the engine has enough cycles
        if len(data) >= seq_length:
            for i in range(len(data) - seq_length + 1):
                seqs.append(data[i : i + seq_length])
                if 'RUL' in group.columns:
                    labels.append(rul[i + seq_length - 1])
    return np.array(seqs), np.array(labels)

# Generate 3D data (Window size = 30 flights)
seq_length = 30
X_train_3D, y_train_3D = generate_sequences(train, seq_length, features)

# 6. Save the final datasets to disk for the rest of the team
train.to_csv('train_clean_2D.csv', index=False)
np.save('X_train_3D.npy', X_train_3D)
np.save('y_train_3D.npy', y_train_3D)

print(f"✅ Preprocessing Complete!")
print(f"2D Data Shape (ANN): {train.shape}")
print(f"3D Data Shape (CNN/LSTM): {X_train_3D.shape}")
