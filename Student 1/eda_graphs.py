import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading clean data for EDA...")
# Load the dataset Student 1 just cleaned
df = pd.read_csv('train_clean_2D.csv')

# Set visual style
sns.set_theme(style="darkgrid")

# --- GRAPH 1: Sensor Degradation for Engine #1 ---
engine_id = 1
engine_data = df[df['id'] == engine_id]

# Selecting a few sensors known to show strong degradation in FD001
sensors_to_plot = ['s2', 's3', 's4', 's11', 's15', 's17']

plt.figure(figsize=(12, 6))
for sensor in sensors_to_plot:
    plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)

plt.title(f'Sensor Degradation Trends Over Time (Engine {engine_id})', fontsize=14, fontweight='bold')
plt.xlabel('Cycles (Flights)', fontsize=12)
plt.ylabel('Normalized Sensor Value', fontsize=12)
plt.axvline(x=engine_data['cycle'].max(), color='r', linestyle='--', label='Engine Failure')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('degradation_trend.png')
print("✅ Saved Graph 1: degradation_trend.png")

# --- GRAPH 2: Piecewise RUL Target Curve ---
plt.figure(figsize=(10, 5))
plt.plot(engine_data['cycle'], engine_data['RUL'], color='purple', linewidth=3)
plt.title(f'Piecewise RUL Curve (Engine {engine_id})', fontsize=14, fontweight='bold')
plt.xlabel('Cycles (Flights)', fontsize=12)
plt.ylabel('Remaining Useful Life (RUL)', fontsize=12)
plt.tight_layout()
plt.savefig('rul_curve.png')
print("✅ Saved Graph 2: rul_curve.png")

# --- GRAPH 3: Sensor Correlation Heatmap ---
plt.figure(figsize=(12, 10))
# Calculate correlation with RUL
correlation = df.drop(columns=['id', 'cycle']).corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Sensor Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("✅ Saved Graph 3: correlation_heatmap.png")

print("\n🎉 Student 1's Job is 100% Complete!")
print("Check your folder for the three PNG images.")
