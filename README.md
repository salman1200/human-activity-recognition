# Human Activity Data Clustering Analysis

## Introduction

This project aims to analyze and cluster human activity data collected from smartphone sensors. The dataset contains readings from accelerometers and gyroscopes for various activities such as walking, walking upstairs, walking downstairs, sitting, standing, and laying. We apply unsupervised machine learning techniques like clustering to group similar activities together based on patterns in the sensor data.

The objective is to discover underlying structures or patterns in the sensor data that differentiate between different human activities. By clustering similar activities together, we can gain insights into how sensor readings vary across different activities and potentially identify common characteristics or features that distinguish one activity from another.

## Data Processing

### Checking and Removing Null Values

- **Purpose**: To ensure data quality and integrity, null values were checked for and handled appropriately.
- **Rationale**: Null values can adversely affect the performance of machine learning algorithms, including clustering. By checking for and removing null values, we ensure that the dataset is clean and complete before further processing. This step helps prevent errors and biases in the analysis caused by missing data.

### Removing Duplicated Records

- **Purpose**: To eliminate redundancy and ensure each record in the dataset is unique.
- **Rationale**: Duplicated records can skew the results of clustering analysis by artificially inflating the importance of certain data points. Removing duplicated records ensures that each data point contributes only once to the analysis, leading to more accurate clustering results and reducing computational overhead.

### Standardization

- **Purpose**: Standardization was applied to scale the data and bring all features into the same scale.
- **Rationale**: Standardization is important for clustering algorithms, as it ensures that all features contribute equally to the clustering process. By scaling the features, we prevent features with larger scales from dominating the distance calculations, leading to more balanced clustering results. This step improves the performance and convergence of clustering algorithms.

### Checking Dataset Balance

- **Purpose**: To assess whether the dataset is balanced or imbalanced in terms of class distribution.
- **Rationale**: Imbalanced datasets, where one class is significantly more prevalent than others, can bias the results of clustering analysis. It's important to check the distribution of class labels or target variables to understand the underlying data distribution and potential biases. By identifying imbalances early in the process, appropriate sampling strategies or class weighting techniques can be applied to mitigate the impact of imbalanced data on clustering results.

![Dataset Balance Visualization](/dataset_balance_image.png)

### Exporting Preprocessed Data to CSV Files

- **Purpose**: To create a snapshot of the preprocessed data for future reference and reproducibility.
- **Rationale**: Exporting preprocessed data to CSV files allows for easy storage, sharing, and reuse of the cleaned dataset. It serves as a record of the preprocessing steps applied to the data, facilitating reproducibility and transparency in the analysis. Additionally, having preprocessed data stored in CSV files enables seamless integration with other tools and platforms for further analysis or collaboration.

### Principal Component Analysis (PCA)

- **Purpose**: PCA was applied to perform dimensionality reduction on the raw dataset.
- **Rationale**: The raw dataset contained 561 features, which would make clustering computationally expensive and inefficient. By applying PCA, the dataset was transformed into a lower-dimensional space while retaining most of the information. This helped reduce the computational complexity of both the clustering algorithms while still capturing the maximum variance in the data. PCA transformed the 561 features into 151 principal components, which captured maximum variance in the data while reducing dimensionality significantly. This made the clustering algorithms more efficient.

### Train-Test Split

- **Purpose**: The dataset was divided into train and test splits for model evaluation.
- **Rationale**: Splitting the dataset into train and test sets allows for the evaluation of clustering models on unseen data. The train set is used to train the clustering model, while the test set is used to assess the generalization performance of the model. This helps to avoid overfitting and provides an estimate of how well the clustering model will perform on new data.

## Modeling

Two clustering algorithms were utilized for the analysis: K-Means and DBSCAN.

### K-Means

- The K-Means algorithm was applied to the PCA-transformed training split as well as without PCA set with the variable number of clusters (K).

### DBSCAN

- DBSCAN was run on the training split using multiple epsilon values between 0.1 to 1.0 and min_samples parameter.

Both models were tested using the training data to identify clusters within the dataset.

### Fine-Tuning of K-Means

The optimal number of clusters (K) for K-Means was determined using the elbow method:

- The elbow point, indicating a significant decrease in within-cluster sum of squares, was observed at 4 clusters. This suggests that 4 is the ideal number of clusters for K-Means.

![K-Means Elbow Method](path/to/kmeans_elbow_plot.png)

### Fine-Tuning of DBSCAN

The optimal epsilon value for DBSCAN was found through experimentation:

- Multiple values of epsilon were tested, and the silhouette statistic was calculated at each step.
- An elbow was observed in the silhouette statistic plot around an epsilon value of 0.3.
- The min_samples parameter was set to 5 based on experimentation.

Both models could identify the 6 true classes present in the dataset very accurately on the test split, with adjusted rand scores greater than 0.9.

![DBSCAN Fine-Tuning](path/to/dbscan_fine_tuning_plot.png)

## Data Processing Steps for Fine Tuning & Visualization

### Dimensionality reduction using PCA

PCA was applied to reduce the dimensionality of the dataset to only 2 before clustering. This helps in improving clustering performance and reducing computational complexity while covering 99% of the data variance. The difference can be seen in the figure given below that the silhouette and Calinski score are significantly better in PCA variant.

![PCA Comparison](path/to/pca_comparison_plot.png)

## Cluster Visualizations

### KMeans with PCA

![KMeans with PCA](path/to/kmeans_pca_plot.png)

### KMeans without PCA

![KMeans without PCA](path/to/kmeans_no_pca_plot.png)

### DBSCAN with PCA

![DBSCAN with PCA](path/to/dbscan_pca_plot.png)

### DBSCAN without PCA

![DBSCAN without PCA](path/to/dbscan_no_pca_plot.png)

## Dimensionality reduction and visualizations

### High Dimensionality and Impractical Visualization

With 561 features, visualizing pair plots or scatter plots in the original feature space became impractical and computationally expensive. The sheer number of dimensions makes it challenging to represent the data in a comprehensible manner, so we converted it to only 2 clusters using the principal component.

### Limited Human Perception and Overplotting perception is limited, and it becomes increasingly difficult to interpret data in high-dimensional spaces. Additionally, without dimensionality reduction, plotting data points directly in their original feature space can lead to overplotting, where data points overlap and hinder each other, making it challenging to collect meaningful patterns or clusters.

### Computational Complexity and Time-Consuming Process

Visualizing high-dimensional data directly requires significant computational resources and time. Generating pair plots or visualizations for 561 features would be computationally intensive and time-consuming, hindering the efficiency of the analysis.

## Conclusion

The clustering analysis of human activity data from smartphone sensors aimed to group similar activities together based on patterns in accelerometer and gyroscope readings. The identified clusters represent distinct patterns of sensor data associated with different human activities, such as walking, walking upstairs, walking downstairs, sitting, standing, and laying. Each cluster captured the unique sensor signatures corresponding to specific activities, allowing for the differentiation and classification of these activities based on their sensor data patterns.
