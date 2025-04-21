# cnn_classification.R
# --------------------------------------------------
# PROJECT: CNN-Based Classification of Columbia Images
#
# OBJECTIVE:
#  - Use the 10×10 grid features (derived from image partitioning) as input.
#  - Reshape these predictors to form a 10×10 image (with 1 channel).
#  - Build a Convolutional Neural Network (CNN) to classify images as indoor (0) or outdoor (1).
#  - Evaluate performance via accuracy, sensitivity, specificity, and ROC analysis.
#
# AUTHOR: Your Name
# DATE: Today's Date
# --------------------------------------------------

#library(keras)
#library(tensorflow)
library(reticulate)

# -------------------------------
# Part 1: Data Preparation
# -------------------------------
# Assume X_train and X_test are already obtained as matrices of predictors (with 100 columns).
# y_train and y_test are the binary labels.
# For this example, we assume that X_train and X_test have dimensions: 
#   (# training samples, 100) and (# test samples, 100) respectively.
# We now reshape these matrices into 10x10 "images" with 1 channel.

# Ensure that our predictors are numeric matrices.
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.numeric(y_train)
y_test <- as.numeric(y_test)

n_train <- nrow(X_train)
n_test <- nrow(X_test)

# Reshape the data so that each sample becomes a 10x10 image with one channel.
X_train_array <- array(X_train, dim = c(n_train, 10, 10, 1))
X_test_array <- array(X_test, dim = c(n_test, 10, 10, 1))

cat("Dimensions of X_train_array:", dim(X_train_array), "\n")
cat("Dimensions of X_test_array:", dim(X_test_array), "\n")

# -------------------------------
# Part 2: CNN Model Construction
# -------------------------------
# CNN architecture explanation:
# - layer_conv_2d: Applies a set of filters (kernels) across the input image.
#   -- "filters": Number of kernels (here 32 or 64) to extract features.
#   -- "kernel_size": Size of each filter (3x3 here).
#   -- "padding" is set to "same" to preserve the spatial dimensions (utilizes zero-padding).
#   -- Strides default to 1, ensuring each pixel is examined.
# - layer_max_pooling_2d: Downsamples the output, reducing spatial dimensions while preserving key features.
# - Dropout layers: Regularize the model by randomly setting a fraction of inputs to zero, reducing overfitting.
# - layer_flatten: Transforms the 2D feature maps into a 1D vector.
# - Fully Connected (Dense) layers: The feedforward layers that combine features to produce the final classification output.
# - Activation functions like ReLU introduce non-linearity, crucial for modeling complex relationships.
# These characteristics help the network to learn robust features and improve sensitivity and specificity.

model_cnn <- keras_model_sequential() %>%
  # First convolutional block
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(10,10,1),
                padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%  # Reduces the spatial dimensions by half (from 10x10 to 5x5)
  layer_dropout(rate = 0.25) %>%
  
  # Second convolutional block
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                activation = "relu",
                padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%  # Further reduces spatial dimensions (from 5x5 to 3x3 approx.)
  layer_dropout(rate = 0.25) %>%
  
  # Flatten and feed into dense (fully connected) layers.
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%  # Feedforward layer: increasing its dimension improves capacity.
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")      # Sigmoid output for binary classification.

# Compile the model.
model_cnn %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

model_cnn %>% summary()

# -------------------------------
# Part 3: CNN Model Training
# -------------------------------
history_cnn <- model_cnn %>% fit(
  x = X_train_array, 
  y = y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 2
)

# -------------------------------
# Part 4: Evaluation Metrics
# -------------------------------
# Evaluate the CNN on the test set.
score_cnn <- model_cnn %>% evaluate(X_test_array, y_test, verbose = 0)
cat("CNN Test Accuracy: ", score_cnn[[2]]*100, "%\n")

# Make predictions on the test set.
pred_prob_cnn <- model_cnn %>% predict(X_test_array)
pred_class_cnn <- ifelse(pred_prob_cnn > 0.5, 1, 0)

# Compute the confusion matrix.
conf_matrix_cnn <- table(Predicted = pred_class_cnn, Actual = y_test)
print("CNN Confusion Matrix:")
print(conf_matrix_cnn)

# Calculate Sensitivity and Specificity:
# Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP)
TP <- conf_matrix_cnn["1", "1"]
TN <- conf_matrix_cnn["0", "0"]
FP <- conf_matrix_cnn["1", "0"]
FN <- conf_matrix_cnn["0", "1"]

sensitivity_cnn <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
specificity_cnn <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)

cat(sprintf("CNN Sensitivity: %.2f%%\n", sensitivity_cnn * 100))
cat(sprintf("CNN Specificity: %.2f%%\n", specificity_cnn * 100))

# -------------------------------
# Part 5: ROC Curve and AUC Calculation
# -------------------------------
library(pROC)
roc_cnn <- roc(y_test, as.vector(pred_prob_cnn))
auc_cnn <- auc(roc_cnn)
cat(sprintf("CNN AUC: %.2f\n", auc_cnn))

# Plot ROC curve.
plot(roc_cnn, col = "blue", main = "ROC Curve for CNN Classifier")
legend("bottomright", legend = sprintf("AUC = %.2f", auc_cnn), col = "blue", lwd = 2)
