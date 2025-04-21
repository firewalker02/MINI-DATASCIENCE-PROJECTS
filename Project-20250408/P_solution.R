# final_project_solution.R
# ------------------------------------------------------------
# PROJECT: Classification of Columbia Campus Images Using Grid Partitioning
#
# OBJECTIVE:
#  - Partition each JPEG image into a 10x10 grid.
#  - For each grid cell, compute the median pixel intensity (averaged over color channels).
#    This results in 100 predictors for each image.
#  - Build two models:
#       (1) A logistic regression classifier.
#       (2) A neural network classifier using the neuralnet package.
#  - Evaluate the methods using overall accuracy, sensitivity, specificity, and ROC/AUC.
#
# REFERENCES:
#  - ADVENT Technical Report #205-2004-5 (Ng et al., Feb 2005)
#  - Additional scene classification literature (e.g., MIT Places dataset)
#
# 
# ------------------------------------------------------------

# -------------------------------
# Part 0: Load Libraries
# -------------------------------
if (!require("jpeg")) install.packages("jpeg", repos="http://cran.us.r-project.org")
if (!require("tidyverse")) install.packages("tidyverse", repos="http://cran.us.r-project.org")
if (!require("neuralnet")) install.packages("neuralnet", repos="http://cran.us.r-project.org")
if (!require("pROC")) install.packages("pROC", repos="http://cran.us.r-project.org")

library(jpeg)       # For reading JPEG images.
library(tidyverse)  # For data manipulation.
library(neuralnet)  # For the feed-forward neural network classifier.
library(pROC)       # For ROC analysis.

# -------------------------------
# Part 1: Data Preparation
# -------------------------------
# Read metadata file. This CSV should have columns: name (filename) and category.
pm <- read.csv("photoMetaData.csv")
n <- nrow(pm)

# Define binary response: for instance, 1 if outdoor image, 0 if indoor.
# Here we assume outdoor images are those with category "outdoor-day". Adjust as needed.
y <- as.numeric(pm$category == "outdoor-day")

# Function to partition an image into a 10x10 grid and compute median intensity.
# For color images, compute median per channel then average across channels.
extract_features <- function(image_path) {
  img <- tryCatch(readJPEG(image_path), error = function(e) { 
    message(sprintf("Error reading file: %s", image_path))
    return(NULL)
  })
  if (is.null(img)) return(rep(NA, 100))
  
  # Get image dimensions.
  dims <- dim(img)  # Expected: (height, width, 3)
  if (length(dims) != 3) {
    # If grayscale, replicate to 3 channels.
    img <- array(rep(img, 3), dim = c(dims[1], dims[2], 3))
    dims <- dim(img)
  }
  
  grid_rows <- 10
  grid_cols <- 10
  height <- dims[1]
  width  <- dims[2]
  cell_height <- floor(height / grid_rows)
  cell_width  <- floor(width / grid_cols)
  features <- numeric(100)
  
  idx <- 1
  for (i in 0:(grid_rows - 1)) {
    for (j in 0:(grid_cols - 1)) {
      row_start <- i * cell_height + 1
      row_end   <- (i + 1) * cell_height
      col_start <- j * cell_width + 1
      col_end   <- (j + 1) * cell_width
      cell <- img[row_start:row_end, col_start:col_end, ]
      # Compute median for each channel and average.
      medians <- apply(cell, 3, median)
      features[idx] <- mean(medians)
      idx <- idx + 1
    }
  }
  return(features)
}

# Build feature matrix X for all images.
X <- matrix(NA, ncol = 100, nrow = n)
for (j in 1:n) {
  img_path <- file.path("columbiaImages", pm$name[j])
  feats <- extract_features(img_path)
  X[j, ] <- feats
  if(j %% 10 == 0) cat(sprintf("%03d / %03d images processed\n", j, n))
}

# Remove any images for which features could not be extracted.
valid_idx <- complete.cases(X)
X <- X[valid_idx, ]
y <- y[valid_idx]
pm <- pm[valid_idx, ]

# Normalize predictors.
X <- scale(X)

# Partition data: 70% training, 30% testing.
set.seed(123)
train_idx <- sample(1:nrow(X), size = floor(0.7 * nrow(X)))
X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]










# -------------------------------
# Part 2: Logistic Regression Classification (Baseline)
# -------------------------------
# -------------------------------
# Part 2: Logistic Regression Classification (Baseline)
# -------------------------------
# Combine predictors and response into a data frame for glm.
train_df <- as.data.frame(X_train)
colnames(train_df) <- paste0("X", 1:ncol(X_train))
train_df$y <- y_train

# Fit logistic regression.
logistic_model <- glm(y ~ ., data = train_df, family = binomial)
summary(logistic_model)

# Predict probabilities on test set.
test_df <- as.data.frame(X_test)
colnames(test_df) <- paste0("X", 1:ncol(X_test))
pred_prob_logistic <- predict(logistic_model, test_df, type = "response")
pred_class_logistic <- ifelse(pred_prob_logistic > 0.5, 1, 0)

# Compute confusion matrix and accuracy.
conf_matrix_logistic <- table(Predicted = pred_class_logistic, Actual = y_test)
logistic_accuracy <- mean(pred_class_logistic == y_test) * 100

cat(sprintf("Logistic Regression Accuracy: %.2f%%\n", logistic_accuracy))

# Calculate sensitivity and specificity.
TP <- conf_matrix_logistic["1", "1"]
TN <- conf_matrix_logistic["0", "0"]
FP <- conf_matrix_logistic["1", "0"]
FN <- conf_matrix_logistic["0", "1"]
sensitivity_logistic <- TP / (TP + FN) * 100
specificity_logistic <- TN / (TN + FP) * 100
cat(sprintf("Sensitivity: %.2f%%, Specificity: %.2f%%\n", sensitivity_logistic, specificity_logistic))

#roc_logistic <- roc(y_test, pred_prob_logistic)
#auc_logistic <- auc(roc_logistic)
#cat(sprintf("Logistic Regression AUC: %.2f%\n", auc_logistic *10))
library(pROC)

# For Logistic Regression:
# Create the ROC object.
roc_logistic <- pROC::roc(response = y_test, predictor = pred_prob_logistic)
auc_logistic <- pROC::auc(roc_logistic)
cat(sprintf("Logistic Regression AUC: %.2f%%\n", auc_logistic *100))

# Plot using the specialized plot.roc() function.
plot.roc(roc_logistic, col = "blue", main = "ROC: Logistic Regression")

# -------------------------------
# Part3: Neural Networks
# -------------------------------



#
# Ensure that predictors are in a data frame and y is appended as a column.
# -------------------------------
train_df <- as.data.frame(X_train)
colnames(train_df) <- paste0("X", 1:ncol(X_train))
train_df$y <- as.numeric(y_train)

test_df <- as.data.frame(X_test)
colnames(test_df) <- paste0("X", 1:ncol(X_test))
test_df$y <- as.numeric(y_test)

# -------------------------------
# Build and Train the Neural Network using the neuralnet package.
# In this example, we use two hidden layers with 128 and 64 neurons.
# -------------------------------
nn_formula <- as.formula(paste("y ~", paste(colnames(train_df)[1:ncol(X_train)], collapse = " + ")))

set.seed(123)
nn_model <- neuralnet(nn_formula,
                      data = train_df,
                     hidden = c(16, 16, 16), #16, 16 max for now
                     linear.output = FALSE, # for classification
                     stepmax = 1e6,
                      lifesign = "minimal")



# Plot the neural network (optional)
plot(nn_model)

# -------------------------------
# Evaluate the Neural Network on the Test Set
# -------------------------------
nn_predictions <- neuralnet::compute(nn_model, test_df[, colnames(test_df) != "y"])
predicted_probabilities <- nn_predictions$net.result  # Predicted probabilities
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Create a confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = test_df$y)
print("Neural Network Confusion Matrix:")
print(conf_matrix)

# Calculate sensitivity and specificity:
# Sensitivity = TP / (TP + FN)
# Specificity = TN / (TN + FP)
TP <- conf_matrix["1","1"]
TN <- conf_matrix["0","0"]
FP <- conf_matrix["1","0"]
FN <- conf_matrix["0","1"]
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
cat(sprintf("Sensitivity: %.2f%%\n", sensitivity * 100))
cat(sprintf("Specificity: %.2f%%\n", specificity * 100))

# Calculate overall accuracy.
accuracy <- mean(predicted_classes == test_df$y)
cat(sprintf("Neural Network Test Accuracy: %.2f%%\n", accuracy * 100))

# -------------------------------
# ROC Curve Calculation using the pROC package
# -------------------------------
#roc_obj <- roc(test_df$y, as.vector(predicted_probabilities))
#auc_value <- auc(roc_obj)
#cat(sprintf("Neural Network AUC: %.2f%%\n", auc_value * 100))

# Plot ROC curve.
roc_nn <- pROC::roc(response = y_test, predictor = as.vector(predicted_probabilities))

auc_value <- auc(roc_obj)
cat(sprintf("Neural Network AUC: %.2f%%\n", auc_value * 100))
plot.roc(roc_nn, col = "red", main = "ROC: Neural Network")


# Save the results to a file.
save(final_results, file = "final_results.Data")
print("Final results saved to final_results.RData")

