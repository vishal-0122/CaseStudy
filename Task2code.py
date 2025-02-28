#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("mcdonalds.csv")
df.columns


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


MD_x = df.iloc[:,:11].copy()


# In[8]:


MD_x = (MD_x == "Yes").astype(int)


# In[9]:


# Rounding off upto 2 decimal places
col_means = MD_x.mean().round(2)
print(col_means)


# In[10]:


# Implement PCA
from sklearn.decomposition import PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)


# In[11]:


std_dev = np.sqrt(pca.explained_variance_)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


# In[12]:


summary_df = pd.DataFrame({
    "Standard Deviation": np.round(std_dev, 5),
    "Proportion of Variance": np.round(explained_variance, 5),
    "Cumulative Proportion": np.round(cumulative_variance, 5)
}, index=[f"PC{i+1}" for i in range(len(std_dev))])
print(summary_df)


# In[13]:


print("Standard deviations:")
print(np.round(std_dev, 1))

print("Rotation (n x k) = (11 x 11):")

loadings = pd.DataFrame(pca.components_.T, columns = [f'PC{i+1}' for i in range(11)], index = MD_x.columns)
print(loadings.round(2))


# In[14]:


# Prinicipal Component Analysis of fast food dataset
pca_scores = MD_pca[:, :2]

#Plotting PCA scores
plt.figure(figsize = (10,8))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], color = 'grey', alpha = 0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.axhline(0, color = 'black', linestyle = 'dashed', linewidth = 1)
plt.axvline(0, color = 'black', linestyle = 'dashed', linewidth = 1)

#Plotting the principal axes
loadings = pca.components_.T[:, :2]

for i, feature in enumerate(MD_x.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color = 'red', alpha = 0.7, head_width = 0.02)
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color = 'red', fontsize = 10)

plt.grid(True)
plt.show()


# In[15]:


#implement kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(1234)

scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

cluster_results = {}

for k in range(2, 9):
    kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 1234)
    cluster_lables = kmeans.fit_predict(MD_x_scaled)
    cluster_results[k] = cluster_lables

MD_km28 = pd.DataFrame(cluster_results)
print(MD_km28.head())

    


# In[16]:


# Compute within-cluster sum of squares (WCSS) for each k
wcss = []
for k in range(1, 9):  # 2 to 8 clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x_scaled)
    wcss.append(kmeans.inertia_)

max_wcss = max(wcss)
yticks_interval = 1500  # Adjust interval as per your output
yticks = np.arange(0, max_wcss + yticks_interval, yticks_interval)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.bar(range(1, 9), wcss, color = 'grey', edgecolor = 'black')
plt.xlabel("Number of Clusters (Segments)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Cluster Evaluation: WCSS for Different Cluster Counts")
plt.xticks(range(1, 9))
plt.yticks(yticks)
plt.grid(axis = 'y', linestyle = '--', aplha = 0.7)

plt.show()


# In[17]:


import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score

np.random.seed(1234)

# Define clustering parameters
k_values = range(2, 9)  # Number of clusters from 2 to 8
n_rep = 10  # Number of repetitions per cluster size
n_boot = 100  # Number of bootstrap samples

# Store ARI results
ari_results = []

for k in k_values:
    boot_labels = []
    
    for _ in range(n_rep):
        # Partially bootstrap (80% of the data) to preserve structure
        boot_sample = resample(MD_x_scaled, replace=True, n_samples=int(0.8 * len(MD_x_scaled)), random_state=np.random.randint(0, 10000))
        
        # Fit KMeans on the bootstrapped sample
        kmeans = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=1234)
        kmeans.fit(boot_sample)
        
        # Store clustering labels
        boot_labels.append(kmeans.predict(MD_x_scaled))  # Predict on full data for comparison

    # Compute ARI pairwise across different clusterings
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            ari = adjusted_rand_score(boot_labels[i], boot_labels[j])
            ari_results.append({"Number of Segments": k, "ARI": ari})

# Convert results to DataFrame
ari_df = pd.DataFrame(ari_results)

# Check ARI values
print("First few rows of ARI DataFrame:")
print(ari_df.head())
print("ARI Min:", ari_df["ARI"].min(), "ARI Max:", ari_df["ARI"].max())

# Plot the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Number of Segments", y="ARI", data=ari_df)

# Set Y-axis range from 0 to 1
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

# Labels and Title
plt.xlabel("Number of Segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Global Stability of Clustering (Bootstrapped ARI)")

# Show the plot
plt.show()


# In[16]:


MD_x_scaled


# In[17]:


from sklearn.metrics import pairwise_distances

# Ensure MD_x_scaled is a DataFrame
if isinstance(MD_x_scaled, np.ndarray):
    MD_x_scaled = pd.DataFrame(MD_x_scaled, columns=[f"Feature_{i}" for i in range(MD_x_scaled.shape[1])])

# Apply KMeans clustering for 4 segments
kmeans_4 = KMeans(n_clusters=4, n_init=20, max_iter=300, random_state=1234)
MD_x_scaled["Cluster"] = kmeans_4.fit_predict(MD_x_scaled)

# Compute pairwise similarity (1 - normalized distance)
dist_matrix = pairwise_distances(MD_x_scaled.drop("Cluster", axis=1), metric="euclidean")
similarity_matrix = 1 - (dist_matrix / np.max(dist_matrix))

# Create a DataFrame to store similarity values per cluster
similarity_data = []
for cluster in range(4):
    cluster_indices = np.where(MD_x_scaled["Cluster"] == cluster)[0]
    cluster_similarities = similarity_matrix[np.ix_(cluster_indices, cluster_indices)].flatten()
    similarity_data.extend([(cluster + 1, sim) for sim in cluster_similarities])

# Convert to DataFrame for plotting
similarity_df = pd.DataFrame(similarity_data, columns=["Cluster", "Similarity"])

# Plot the Gorge Plot using Seaborn's FacetGrid
g = sns.FacetGrid(similarity_df, col="Cluster", col_wrap=2, sharex=True, sharey=True, height=4)
g.map(sns.histplot, "Similarity", bins=20, color="black", kde=False)

# Adjust aesthetics
g.set_axis_labels("Similarity", "Percent of Total")
g.set_titles("Cluster {col_name}")
plt.xlim(0.2, 0.8)  # Match the x-axis range in the R plot
plt.show()


# In[19]:


from sklearn.metrics import pairwise_distances


# Assuming df is your main dataset, and we extract relevant features into MD_x
MD_x = np.random.rand(100, 5)  # Replace with actual features from your dataset

# Perform k-means clustering with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
MD_k4 = kmeans.fit(MD_x)

# Function to compute Segment-Level Stability Within solutions (SLSW)
def slsw(X, labels, n_clusters):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    stability = {i: [] for i in range(n_clusters)}
    
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points, [centroids[i]])
            stability[i] = 1 - distances.flatten()  # Invert distance for stability
    
    return stability

# Compute segment-level stability
labels = MD_k4.labels_
stability_dict = slsw(MD_x, labels, k)

# Convert stability data into a format suitable for Seaborn boxplot
stability_data = []
for segment, stability_values in stability_dict.items():
    for value in stability_values:
        stability_data.append((segment + 1, value))  # Segment numbers start from 1

stability_df = pd.DataFrame(stability_data, columns=["Segment Number", "Segment Stability"])

# Plot the boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x="Segment Number", y="Segment Stability", data=stability_df)
plt.ylim(0, 1)
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability")
plt.title("Segment-Level Stability (Boxplot)")
plt.show()


# In[20]:


print(df.columns)
df['Like'].value_counts(ascending=True)


# In[21]:


# Convert 'Like' column to numeric, then apply transformation
df["Like.n"] = 6 - pd.to_numeric(df["Like"], errors="coerce")

# Generate frequency table
like_n_table = df["Like.n"].value_counts().sort_index()

# Display the frequency table
print(like_n_table)


# In[22]:


# Extract first 11 column names and create the formula string
formula_str = "Like.n ~ " + " + ".join(df.columns[:11])

print(formula_str)


# In[23]:


import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

# Compute distance matrix for transposed data
distance_matrix = pdist(MD_x.T)  # Equivalent to dist(t(MD.x)) in R

# Perform hierarchical clustering
MD_vclust = sch.linkage(distance_matrix, method="ward")  # Ward's method (default in hclust)

# Display clustering result
print(MD_vclust)


# In[24]:


cluster_labels = MD_k4.labels_  # Get cluster labels

# Compute hierarchical clustering order (equivalent to MD.vclust$order in R)
distance_matrix = pdist(MD_x.T)  # Distance on transposed data
MD_vclust = sch.linkage(distance_matrix, method="ward")
order = sch.leaves_list(MD_vclust)  # Get the ordering of features

# Create a heatmap/barchart representation
plt.figure(figsize=(10, 6))
sns.heatmap(MD_x[:, order].T, cmap="coolwarm", annot=False, cbar=True)

plt.xlabel("Observations")
plt.ylabel("Reordered Features")
plt.title("Clustered Barchart Heatmap")
plt.show()


# In[25]:


# Perform PCA
pca = PCA(n_components=2)  # Project onto 2 principal components
MD_pca = pca.fit_transform(MD_x)

# Perform k-means clustering (assuming k=4, from MD.k4)
kmeans = KMeans(n_clusters=4, random_state=1234, n_init=10)
cluster_labels = kmeans.fit_predict(MD_x)

# Plot the PCA projection with cluster labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=MD_pca[:, 0], y=MD_pca[:, 1], hue=cluster_labels, palette="tab10", s=50)

# Customize labels
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Visualization using PCA")

# Optional: Draw projection axes (equivalent to projAxes in R)
for i, (comp_x, comp_y) in enumerate(pca.components_.T):
    plt.arrow(0, 0, comp_x, comp_y, color="black", alpha=0.5, head_width=0.02, label=f"Feature {i+1}")

plt.legend(title="Clusters")
plt.grid(True)
plt.show()


# In[ ]:




