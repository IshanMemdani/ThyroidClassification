import pandas as pd

# Generate a sample dataset with columns for thyroid disorder prediction
data = pd.DataFrame({
    'T3': [1.2, 2.5, 3.0, 0.8, 2.2],  # Triiodothyronine levels
    'T4': [120, 115, 130, 100, 110],  # Thyroxine levels
    'TSH': [1.5, 2.0, 1.8, 3.0, 2.4],  # Thyroid Stimulating Hormone levels
    'age': [45, 50, 35, 60, 40],  # Age of the person
    'gender': ['female', 'male', 'female', 'male', 'female'],  # Gender
    'medical_history': ['none', 'diabetes', 'none', 'hypertension', 'none']  # Medical history
})

# Save the dataset as a CSV file (optional, for reference)
data.to_csv('sample_thyroid_data.csv', index=False)

# Print the dataset (optional, for debugging)
print(data)