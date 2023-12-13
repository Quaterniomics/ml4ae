#MTAER: Methylation-based Tissue Age Estimation in R ( software developed by Joseph Anthony Campagna TUID915291684 )
# Biostatistics with R
# Date: 12/12/2023
# Time: 11:45 PM ET
# Purpose: Processing and analyzing GTEx muscle methylation data

# Required libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(umap)
library(rgl)
library(glmnet)

# Load and preprocess annotation data
anno_data_path <- 'GTEx_Muscle.anno.csv'
anno_data <- read.csv(anno_data_path, sep = '\t', header = TRUE, stringsAsFactors = FALSE)

# Convert age range strings to lower bound integers
anno_data$age <- as.numeric(sapply(strsplit(as.character(anno_data$age), "-"), `[`, 1))

# Transpose the data and set the first row as column headers
anno_data_transposed <- as.data.frame(t(anno_data))
colnames(anno_data_transposed) <- anno_data_transposed[1, ]
anno_data_transposed <- anno_data_transposed[-1, ]

# Convert character data to numeric where appropriate
anno_data_transposed$age <- as.numeric(sub("-.*", "", anno_data_transposed$age))
anno_data_transposed$Sex <- ifelse(anno_data_transposed$Sex == '1', 1, 0)
anno_data_transposed$smoker_status <- ifelse(anno_data_transposed$smoker_status %in% c('Non-smoker', NA), 0, 1)

# Sort the data by 'Sex' and 'age'
sorted_data <- anno_data_transposed %>% 
  arrange(Sex, age)

# Load methylation data and merge with annotation data
meth_data_path <- 'GTEX_Muscle.meth.csv'
meth_data <- read_csv(meth_data_path, col_types = cols(.default = col_double()))
meth_data_transposed <- as.data.frame(t(meth_data))
colnames(meth_data_transposed) <- meth_data_transposed[1, ]
meth_data_transposed <- meth_data_transposed[-1, ]

# Merge annotation and methylation data
mergeset <- merge(sorted_data, meth_data_transposed, by = "row.names", all = TRUE)

# Convert non-numeric data to numeric and drop NaNs in methylation data
cpg_data <- mergeset[, 12:ncol(mergeset)]  # Assuming CpG data starts from the 12th column
cpg_data[is.na(cpg_data)] <- 0

# Filtering the DataFrame for the two age groups
group1_data <- cpg_data[mergeset$age == 30, ]
group2_data <- cpg_data[mergeset$age == 70, ]

# Perform t-tests and calculate mean differences for each CpG site
cpg_columns <- colnames(cpg_data)
results <- list()

for (col in cpg_columns) {
  group1_values <- group1_data[, col, drop = FALSE]
  group2_values <- group2_data[, col, drop = FALSE]
  
  t_test_result <- t.test(group1_values, group2_values, na.rm = TRUE)
  mean_diff <- mean(group1_values, na.rm = TRUE) - mean(group2_values, na.rm = TRUE)
  
  results[[col]] <- list(mean_diff = mean_diff, p_val = t_test_result$p.value)
}

# Convert results to a data frame and sort by mean_diff
diff_methylation_df <- do.call(rbind, lapply(results, data.frame, stringsAsFactors = FALSE))
diff_methylation_df$CpG_site <- rownames(diff_methylation_df)
rownames(diff_methylation_df) <- NULL
sorted_diff <- diff_methylation_df[order(diff_methylation_df$mean_diff, decreasing = TRUE), ]

# Select top 20 hypermethylated and bottom 20 hypomethylated sites
top20 <- head(sorted_diff, 20)
bottom20 <- tail(sorted_diff, 20)
selected_cpgs <- rbind(top20, bottom20)

# Extracting the selected CpG sites along with patient sample IDs
selected_sites <- selected_cpgs$CpG_site
selected_data <- mergeset[, c("row.names", selected_sites)]

# Filter the mergeset for samples that are either 30 or 70 years old
age_filtered_data <- subset(mergeset, age %in% c(30, 70))

# Extracting only the CpG site columns of interest from the filtered data
selected_data_age_filtered <- age_filtered_data[, c("row.names", selected_sites)]

# Differential methylation heatmap; age-based (30 vs 70)
library(pheatmap)
pheatmap(as.matrix(selected_data_age_filtered[, -1]),
         color = colorRampPalette(c("blue", "white", "red"))(100),
         labels_row = selected_data_age_filtered$`row.names`)

# Bar plot of differential methylation; age-based (30 vs 70)
ggplot(selected_cpgs, aes(x = CpG_site, y = mean_diff)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("CpG Sites") +
  ylab("Mean Difference in Methylation") +
  ggtitle("Differential Methylation Bar Plot")

# Volcano plot of differential methylation; age-based (30 vs 70)
ggplot(selected_cpgs, aes(x = mean_diff, y = -log10(p_val))) +
  geom_point() +
  xlab("Mean Difference in Methylation") +
  ylab("-log10(p-value)") +
  ggtitle("Volcano Plot of Differential Methylation")

# 3D UMAP 

# Assuming 'cpg_data' is a dataframe containing methylation data
# Using the same simulated data as above for illustration

# Applying UMAP
umap_result <- umap(cpg_data, n_components = 3)

# Convert to DataFrame for visualization
embedding_df <- as.data.frame(umap_result$layout)

# Plotting 3D UMAP
plot3d(embedding_df$V1, embedding_df$V2, embedding_df$V3, col = "red", size = 1)
axes3d()
title3d("3D UMAP of Methylation Data", line = 2.5)

# Assuming 'X_train', 'X_test', 'y_train', 'y_test' are your training and test sets
# Load or generate your dataset here

# Define a grid of hyperparameters
grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = c(0.1, 0.5, 1, 5, 10))

# Elastic Net model using caret package
control <- trainControl(method = "cv", number = 5)
enet_model <- train(x = X_train, y = y_train, method = "glmnet",
                    tuneGrid = grid, trControl = control)

# Best hyperparameters
print(paste("Best Alpha:", enet_model$bestTune$alpha))
print(paste("Best Lambda:", enet_model$bestTune$lambda))

# Predict and Evaluate
y_pred <- predict(enet_model, X_test)

# Calculate Mean Squared Error
mse <- mean((y_test - y_pred)^2)
print(paste("Mean Squared Error:", mse))





