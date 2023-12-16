# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the dataset
file_path = 'spotify_songs.csv'
df = pd.read_csv(file_path)

# Convert the release date to timestamp
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'])
df['track_album_release_date'] = df['track_album_release_date'].apply(lambda x: x.timestamp())

# Print the original dataframe shape
print('Original Dataframe shape :', df.shape)

# Separate features (X) and target variable (y)
X = df.drop('playlist_genre', axis=1)
print('Inputs Dataframe shape   :', X.shape)

# Standardize the features
X_mean = X.mean()
X_std = X.std()
Z = (X - X_mean) / X_std

# Calculate the covariance matrix and visualize it
c = Z.cov()
sns.heatmap(c)
plt.show()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(c)
print('Eigen values:\n', eigenvalues)
print('Eigen values Shape:', eigenvalues.shape)
print('Eigen Vector Shape:', eigenvectors.shape)

# Sort eigenvalues and corresponding eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Compute cumulative explained variance
explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
print(explained_var)

# Determine the number of components that explain at least 50% of the variance
n_components = np.argmax(explained_var >= 0.50) + 1
print(n_components)

# Extract the top principal components
u = eigenvectors[:, :n_components]
pca_component = pd.DataFrame(u, index=X.columns, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# Visualize the top principal components
plt.figure(figsize=(5, 7))
sns.heatmap(pca_component)
plt.title('PCA Component')
plt.show()

# Apply PCA to the standardized data
pca = PCA(n_components=5)
pca.fit(Z)
x_pca = pca.transform(Z)
df_pca1 = pd.DataFrame(x_pca, columns=['PC{}'.format(i+1) for i in range(n_components)])

# Visualize songs in a scatter plot based on the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df['playlist_genre'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# Print the principal components
print(pca.components_)

# Encode the target variable
y = df['playlist_genre']
sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Visualize the LDA-transformed data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')

# Train a Random Forest classifier on the LDA-transformed data
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = classifier.predict(X_test)
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)
