import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# Step 1: Load dataset
# =============================
df = pd.read_excel("c:/Users/270374/Downloads/crp.xlsx")  # Change path if needed
print(df.head())

# =============================
# Step 2: Data preprocessing
# =============================
df.dropna(inplace=True)  # Remove missing values
df['Year'] = pd.to_datetime(df['Year'], format='%Y')  # Ensure Year is datetime

# Create Total Crimes column
crime_columns = df.columns[3:]  # All crime categories from column index 3 onward
df['Total_Crimes'] = df[crime_columns].sum(axis=1)

# =============================
# Step 3: EDA Visualizations
# =============================

# 1️⃣ Top 10 Cities by Total Crimes
top_cities = df.groupby('City')['Total_Crimes'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(
    x=top_cities.values, 
    y=top_cities.index, 
    hue=top_cities.index, 
    palette="Reds_r", 
    legend=False
)
plt.title("Top 10 Cities by Total Crimes")
plt.xlabel("Number of Crimes")
plt.ylabel("City")
plt.show()

# 2️⃣ Yearly Crime Trend (Total Crimes)
yearly_trend = df.groupby(df['Year'].dt.year)['Total_Crimes'].sum()
plt.figure(figsize=(8,5))
plt.plot(yearly_trend.index, yearly_trend.values, marker='o', color='blue')
plt.title("Yearly Crime Trend (Total Crimes)")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.show()

# 3️⃣ Heatmap of Feature Correlations
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4️⃣ Distribution of a Specific Crime Type (e.g., Murder)
plt.figure(figsize=(8,5))
sns.histplot(df['Murder'], bins=15, kde=True, color="red")
plt.title("Distribution of Murder Cases")
plt.xlabel("Number of Murders")
plt.ylabel("Frequency")
plt.show()

# =============================
# Step 4: Machine Learning Model
# =============================

# Predict Total Crimes based on population & selected crime categories
X = df[['Population (in Lakhs) (2011)+', 'Murder', 'Kidnapping', 'Crime against women',
        'Crime against children', 'Cyber Crimes']]  # features
y = df['Total_Crimes']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =============================
# Step 5: Model Evaluation
# =============================
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Prediction vs Actual Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Total Crimes")
plt.ylabel("Predicted Total Crimes")
plt.title("Prediction vs Actual")
plt.show()
