import pandas as pd
import torch
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve
import yellowbrick as yb
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#12|12|2023 TAER [Tissue Age Estimator using R language] Beyond R MEAT Package-- Exploring The Epigenetic Landscape of Geroscience & Aging
#provide short description of the code
"""The intent of this Python code, which has equivalent translations in R and Julia (all of which can be found on the github repository), is to provide a simple example of how to perform
 differential methylation analysis using the GTEx dataset. The code is based on the following paper: https://pubmed.ncbi.nlm.nih.gov/24138928/ .
"""
"""
TAER + Machine Learning 101: Predictive Modeling of Biological Age Using Machine Learning on the GTEx Database's Muscle Tissue Methylation Data

Data Source: https://gtexportal.org/home/datasets -- n=47 samples of the vastus lateralis muscle tissue from the GTEx database (https://gtexportal.org/home/).
raw data geometry: 754000x47 matrix of methylation beta values for 754000 CpG sites in 47 samples of the vastus lateralis muscle tissue from the GTEx database (https://gtexportal.org/home/).

Objective: "Analyze the data by performing at least two different statistical analyses using R.
Using two different post hoc analyses or performing two t-tests are NOT considered as
performing two different statistical analyses. The reasons why you choose to perform these
analyses need to be explained and included in the report. The R codes, the results, and the
conclusions must be included as well. If you are performing any hypothesis tests, appropriate
hypotheses also need to be included in the report."
A. Two statistical analyses performed post hoc: 
    I. Elastic Net Regression (L1 and L2 regularization) and t-tests) for Biological Age Prediction [Predictive Machine Learning Model]
    II. Unsupervised Learning-- Classifying Unlabeled Tissue Samples by Tissue Type (K-means clustering) [Unsupervised Machine Learning Model]

B. The reasons why I chose to perform these analyses are as follows:
    I. I am a neuroscientist by training, and I am interested in the epigenetic basis of aging and neurodegenerative diseases. This type of data science is utilized in the longevity startup I work with.
    II. I am interested in the epigenetic basis of aging and neurodegenerative diseases. This type of data science is useful for geroscience, aging research, longevity, regenerative medicine, and precision medicine.
    III. I am entrenched into tech culture + the tech industry, and I am interested in the intersection of tech and biology. I am also interested in the intersection of tech and healthcare.
    IV. I am a hobbyist AI engineer & desired to explore the trending field of 'biological age' prediction using machine learning. I did not have enough time to use a convolutional network w labeled neuroimaging data.
This program is written in R, Python, & Julia language. The R code is the original code, and the Python and Julia code are translations of the R code.

The R code is written in RStudio, and the Python code is written in Jupyter Notebook. The Julia code is written in JuliaPro.
This software was created by Joseph Anthony Campagna, a student of structural informatics and machine learner at Temple University. This software was published on 12/12/2023 at 11:45 PM EST.

Creator: Joe Campagna ( Find me on LinkedIn or @quaterniomics on X/Twitter for education, research, and career opportunities.)
Date of Creation: 12/12/2023
Date of Publication: 12/12/2023
Date of Last Update: 12/12/2023
Version: 1.0.0
License: MIT License
ATTENTION: Any updates after 12/12/2023 to the github repository code and/or markdown README are not reflected of Dr. Ang Sun's Biostatistics R course at Temple University.
"""
# Setting device to GPU if CUDA is available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#check cuda
"""print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_initialized())"""

#Part 1.1: Differential Methylation Analysis of the GTEx Muscle Tissue Methylation Data [Pre-Processing and Exploratory Data Analysis]
# Load and preprocess annotation data
anno_data_path = 'GTEx_Muscle.anno.csv'
anno_data = pd.read_csv(anno_data_path, sep='\t', header=None)
anno_data_transposed = anno_data.transpose()
anno_data_transposed.columns = anno_data_transposed.iloc[0]
anno_data_transposed = anno_data_transposed.drop(anno_data_transposed.index[0])
anno_data_transposed = anno_data_transposed.set_index(anno_data_transposed.columns[0])
anno_data_processed = anno_data_transposed.copy()
anno_data_processed['age'] = anno_data_processed['age'].apply(lambda x: int(x.split('-')[0]))
anno_data_processed['Sex'] = anno_data_processed['Sex'].map({'1': 1, '2': 0})
anno_data_processed['smoker_status'] = anno_data_processed['smoker_status'].map({'Non-smoker': 0, 'Current': 1, 'Former': 1})
anno_data_processed['smoker_status'] = anno_data_processed['smoker_status'].fillna(0)
sorted_data = anno_data_processed.sort_values(by=['Sex', 'age'], ascending=[True, True])

# Load methylation data and merge with annotation data
meth_data_path = 'GTEX_Muscle.meth.csv'
meth_data = pd.read_csv(meth_data_path, sep='\t', header=0)
meth_data_transposed = meth_data.set_index(meth_data.columns[0]).transpose()
mergeset = sorted_data.merge(meth_data_transposed, left_index=True, right_index=True)

# Convert non-numeric data to numeric and drop NaNs
cpg_data = mergeset.iloc[:, mergeset.columns.get_loc('smoker_status') + 1:]
cpg_data = cpg_data.apply(pd.to_numeric, errors='coerce').fillna(0)

#saving cpg_data as a csv file
# Assuming 'cpg_data' is your DataFrame for the part of the project involving R language
cpg_data.to_csv('cpg_data.csv', index=False)

# Filter DataFrame for two age groups and convert to PyTorch tensors
group1_data = cpg_data[mergeset['age'] == 30]
group2_data = cpg_data[mergeset['age'] == 70]
group1_tensor = torch.tensor(group1_data.values, dtype=torch.float32).to(device)
group2_tensor = torch.tensor(group2_data.values, dtype=torch.float32).to(device)

# Perform t-tests and calculate mean differences for each CpG site
cpg_columns = list(meth_data_transposed.columns)
results = []
for col in cpg_columns:
    group1_values = group1_tensor[:, cpg_data.columns.get_loc(col)]
    group2_values = group2_tensor[:, cpg_data.columns.get_loc(col)]
    t_stat, p_val = ttest_ind(group1_values.cpu().numpy(), group2_values.cpu().numpy(), nan_policy='omit')
    mean_diff = group1_values.mean() - group2_values.mean()
    results.append((col, mean_diff.item(), p_val))

# Convert results to a DataFrame and sort by mean_diff
diff_methylation_df = pd.DataFrame(results, columns=['CpG_site', 'mean_diff', 'p_val'])
sorted_diff = diff_methylation_df.sort_values(by='mean_diff', ascending=False)

# Select top 20 hypermethylated and bottom 20 hypomethylated sites
top20 = sorted_diff.head(20)
bottom20 = sorted_diff.tail(20)
selected_cpgs = pd.concat([top20, bottom20])

# Extracting the selected CpG sites along with patient sample IDs
selected_sites = selected_cpgs['CpG_site'].tolist()
selected_data = mergeset[selected_sites]
# Filter the mergeset for samples that are either 30 or 70 years old
age_filtered_data = mergeset[(mergeset['age'] == 30) | (mergeset['age'] == 70)]
# Extracting only the CpG site columns of interest from the filtered data
selected_data_age_filtered = age_filtered_data[selected_cpgs['CpG_site'].tolist()]

#Differential methylation heatmap; age-based (30 vs 70)
# plt.figure(figsize=(12, 8))
# sns.heatmap(selected_data_age_filtered, cmap="coolwarm")
# plt.title("Differential Methylation Heatmap")
# plt.xlabel("CpG Sites")
# plt.ylabel("Patient Samples")
# plt.show()

#Volcano plot of differential methylation; age-based (30 vs 70)
#Each point is annotated with the CpG site name, whereas significant outlier points are color-coded distinctly.
plt.figure(figsize=(12, 8))
plt.scatter(selected_cpgs['mean_diff'], -np.log10(selected_cpgs['p_val']), c='blue')
plt.scatter(selected_cpgs[selected_cpgs['p_val'] < 0.05]['mean_diff'], -np.log10(selected_cpgs[selected_cpgs['p_val'] < 0.05]['p_val']), c='red')
plt.xlabel('Mean Difference in Methylation')
plt.ylabel('-log10(p-value)')
plt.title('Differential Methylation Volcano Plot')
plt.show()

#Part 1.2: Building Elastic Net Regression (L1 and L2 regularization) for Biological Age Prediction [Predictive Machine Learning Model]
# Split data into training and testing sets
X = selected_data
y = age_filtered_data['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net Regression with Grid Search
parameters = {'alpha': [0.1, 0.5, 1, 5, 10], 'l1_ratio': np.arange(0.0, 1.0, 0.1)}
enet = ElasticNet(max_iter=10000)
grid = GridSearchCV(enet, parameters, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid.best_params_)

# Train model with best parameters
best_enet = ElasticNet(**grid.best_params_, max_iter=10000)
best_enet.fit(X_train, y_train)

# Predict and Evaluate
y_pred = best_enet.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Simple way to calculate the mean absolute error, mean squared error, and root mean squared error from a Pandas DataFrame
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

print("Mean Absolute Error:", mae(y_test, y_pred))
print("Mean Squared Error:", mse(y_test, y_pred))
print("Root Mean Squared Error:", rmse(y_test, y_pred))

#Part 2.1: Unsupervised Learning-- Classifying Unlabeled Tissue Samples by Tissue Type (K-means clustering) [Unsupervised Machine Learning Model]
"""Here, we build a simple classifier that utilizes GTEx samples as data points and tissue types as labels. 
We will use the K-means clustering algorithm to cluster the samples into 30 clusters, which is the number of tissue types in the dataset. 
We will then compare the predicted labels to the actual labels to evaluate the performance of the classifier."""

#using the same data as above
# Convert data to PyTorch tensor
data_tensor = torch.tensor(selected_data.values, dtype=torch.float32).to(device)

# K-means clustering
kmeans = KMeans(n_clusters=30, random_state=42)
kmeans.fit(data_tensor.cpu().numpy())
labels = kmeans.labels_

# Evaluate performance
rand_score = adjusted_rand_score(age_filtered_data['tissue_type'], labels)
print("Adjusted Rand Score:", rand_score)

print("'Thank you for using TAER! Please cite this code if you use it in your research. Have a great day! :)' - Joe Campagna, Neuroscience B.S. & Structural Informaticist M.S.")