import os

extract_path = r"C:\Users\anush\Downloads\metadata"  # Change to actual folder name

# List files
files = os.listdir(extract_path)
print(files)
import pandas as pd

# Update the path based on your extracted location
metadata_path = r"C:\Users\anush\Downloads\metadata\metadata"  

# Load the metadata into a DataFrame
df = pd.read_csv(metadata_path, low_memory=False)

# Display the first few rows
print(df.head())
print(df.info())  # Shows column names and data types
print(df.columns) # Lists all column names
print(df.isnull().sum())  # Shows missing values per column
print(df['label'].value_counts())  # Change 'label' to the actual column name


# Fill missing values with median for numerical columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical values with "Unknown"
df.fillna("Unknown", inplace=True)

from sklearn.utils import resample

# Separate FAKE and REAL samples
df_real = df[df["label"] == "REAL"]
df_fake = df[df["label"] == "FAKE"]

# Undersample FAKE to match REAL count
df_fake_downsampled = resample(df_fake, replace=False, n_samples=len(df_real), random_state=42)

# Combine both
df_balanced = pd.concat([df_real, df_fake_downsampled])

print(df_balanced["label"].value_counts())  # Now balanced!

df = pd.get_dummies(df, columns=["audio.@codec_name", "video.@codec_time_base"], drop_first=True)

from sklearn.utils import resample

df_fake = df[df["label"] == "FAKE"]
df_real = df[df["label"] == "REAL"]

df_fake_downsampled = resample(df_fake, 
                               replace=False,  # Don't duplicate
                               n_samples=len(df_real),  # Match real samples
                               random_state=42)

df_balanced = pd.concat([df_fake_downsampled, df_real])
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)  # Shuffle
print(df_balanced["label"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=df_balanced["label"])
plt.title("Balanced Dataset Distribution")
plt.show()












