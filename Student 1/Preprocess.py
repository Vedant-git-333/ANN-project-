import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load the raw data
columns = ['id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
train_full = pd.read_csv('train_FD001.txt', sep=r'\s+', names=columns)
test = pd.read_csv('test_FD001.txt', sep=r'\s+', names=columns)

# 2. Drop the flatline sensors and static settings
drop_cols = ['op3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
train_full.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

# 3. Calculate Piecewise RUL for training data (Cap at 125)
MAX_RUL = 125
train_max_cycle = train_full.groupby('id')['cycle'].max().reset_index()
train_max_cycle.columns = ['id', 'max_cycle']
train_full = train_full.merge(train_max_cycle, on=['id'], how='left')
train_full['RUL'] = train_full['max_cycle'] - train_full['cycle']
train_full['RUL'] = train_full['RUL'].clip(upper=MAX_RUL)
train_full.drop(columns=['max_cycle'], inplace=True)

# 4. Engine-Level Train/Validation Split (80/20)
engine_ids = train_full['id'].unique()
np.random.seed(42) # For reproducibility
np.random.shuffle(engine_ids)

split_idx = int(len(engine_ids) * 0.8)
train_ids, val_ids = engine_ids[:split_idx], engine_ids[split_idx:]

train = train_full[train_full['id'].isin(train_ids)].copy()
val = train_full[train_full['id'].isin(val_ids)].copy()

# 5. Normalize (Fit ONLY on the new 80% train set)
features = train.columns.drop(['id', 'cycle', 'RUL'])
scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])
val[features] = scaler.transform(val[features])
test[features] = scaler.transform(test[features])

# 6. Build the Sliding Window function for CNN/LSTM
def generate_sequences(df, seq_length, feature_cols):
    seqs, labels = [], []
    for engine_id, group in df.groupby('id'):
        data = group[feature_cols].values
        rul = group['RUL'].values if 'RUL' in group.columns else None
        
        if len(data) >= seq_length:
            for i in range(len(data) - seq_length + 1):
                seqs.append(data[i : i + seq_length])
                if rul is not None:
                    labels.append(rul[i + seq_length - 1])
    return np.array(seqs), np.array(labels)

# Generate 3D data (Window size = 30 flights)
seq_length = 30
X_train_3D, y_train_3D = generate_sequences(train, seq_length, features)
X_val_3D, y_val_3D = generate_sequences(val, seq_length, features)

# 7. Save the final datasets
train.to_csv('train_clean_2D.csv', index=False)
val.to_csv('val_clean_2D.csv', index=False)

np.save('X_train_3D.npy', X_train_3D)
np.save('y_train_3D.npy', y_train_3D)
np.save('X_val_3D.npy', X_val_3D)
np.save('y_val_3D.npy', y_val_3D)

print(f"✅ Preprocessing 100% Complete with Validation Split!")
print(f"Training Engines: {len(train_ids)} | Validation Engines: {len(val_ids)}")
print(f"3D Train Shape: {X_train_3D.shape} | 3D Val Shape: {X_val_3D.shape}")
