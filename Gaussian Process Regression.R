                    #################################################################
                    #                                                               #
                    # PREDICTING SUPERCONDUCTING CRITICAL TEMPERATURE USING         #
                    #   GAUSSIAN PROCESS REGRESSION: A COMPARATIVE ANALYSIS         #
                    #                                                               #
                    #################################################################

#---------------------------------------------------------------------------------------------------------------#
# The capability of the Gaussian Process Regression (GPR) in predicting the critical temperature of             #
#  superconductors was assessed. Principal Component Analysis (PCA) and Kernel Principal Component Analysis     #
#  (KPCA) were also explored to assess their impact on the GPR model performance. Furthermore, other models,    #
#  Linear Regression (LR), Random Forest (RF), and Support Vector Regression (SVR) were also used in predicting #
#  the superconductors’ critical temperature.                                                                   #
#---------------------------------------------------------------------------------------------------------------#



#Loading Libraries
library(tidyverse)
library(caret)
library(kernlab)
library(ggplot2)
library(corrplot)
library(randomForest)
library(MASS)
library(e1071)
library(DiceKriging)
library(GGally)
library(rpart)
library(BBmisc)

#Loading Dataset

df = read.csv(r"(C:\Users\eniha\Documents\Coventry\Optimization&Modelling\Coursework\superconductivty+data\train.csv)")

#First few rows
head(df)

#Summary statistics
summary(df)

#Checking for missing values
sum(is.na(df))

#Distribution of the target variable
ggplot(df, aes(x = critical_temp)) +
  geom_histogram(bins = 50, fill = "blue", color="black", alpha = 0.7) +
  labs(title = "Distribution of Critical Temperature", x = "Critical Temperature", y = "Frequency") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))

#Boxplot to check outliers
ggplot(df, aes(x=factor(number_of_elements), y = critical_temp)) +
  geom_boxplot(fill = "blue", alpha = 0.5) +
  labs(title = "Boxplot of Critical Temperature", y = "Critical Temperature") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))



# Identifying highly correlated features (above 0.7 correlation)
df2 = df
high_corr = findCorrelation(cor_matrix, cutoff = 0.7)

# Retaining high-correlation features for correlation matrix
df2 = df2[, high_corr]  

#Heatmap Matrix for correlations above 0.7 
cor_matrix = cor(df2[, -ncol(df)])  # Compute correlation matrix excluding target
corrplot::corrplot(cor_matrix, method = "color", title = "Correlation Matrix Above 0.7", tl.cex = 0.7)


# Density plot for the number of element in each superconductor 
ggplot(df, aes(x = df[,1])) + 
  geom_density(fill = "blue", alpha = 0.5) +
  labs(title = "Density Plot for Number of Elements in Superconductor", x = "Number of Elements", y = "Density") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))


#Pair plots feature relationships
ggpairs(df[, c(1:20, ncol(df))], title="Pairplot for first 20 Features")

ggpairs(df[, c(71:81, ncol(df))], title="Pairplot Last 10 Features")


#PREPROCESSING

#Splitting the dataset
set.seed(25)

# Features and target variables
X = df[, colnames(df) != "critical_temp"]
y = df$critical_temp

#Splitting the X feature and target variables into training and testing 
train_index = createDataPartition(y, p = 0.7, list = FALSE)
X_train = X[train_index, ]
X_test = X[-train_index, ]
y_train = y[train_index]
y_test = y[-train_index]



# Standardizing the features
standard_preprocess = preProcess(X_train, method = c("center", "scale"))

X_train_scaled = predict(standard_preprocess, X_train)
X_test_scaled = predict(standard_preprocess, X_test)



#GPR MODEL
# Training a Gaussian Process Regression model on the scaled data
gpr_model = gausspr(x = as.matrix(X_train_scaled), y = y_train, kernel = "rbfdot")

# GPR Model Predictions
gpr_pred = predict(gpr_model, as.matrix(X_test_scaled))

# GPR Model Evaluation
gpr_mse = mean((y_test - gpr_pred)^2)
gpr_rmse = sqrt(gpr_mse)
gpr_mae = mean(abs(y_test - gpr_pred))
gpr_r2 = cor(y_test, gpr_pred)^2

print(c(GPR_MSE = gpr_mse, GPR_RMSE = gpr_rmse, GPR_MAE = gpr_mae, GPR_R2 = gpr_r2))

# Approximating gpr_pred variance
gpr_var = kernlab::predict(gpr_model, as.matrix(X_test_scaled), type = "probabilities")  
gpr_sd = sqrt(gpr_var)


# Data frame of GPR result for plotting
GPR_df = data.frame(Actual = y_test, Predicted = gpr_pred, 
                        Upper_Bound = gpr_pred + 1.96 * gpr_sd, Lower_Bound = gpr_pred - 1.96 * gpr_sd)

# Plot of predictions with uncertainty bands
ggplot(GPR_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_ribbon(aes(ymin = Lower_Bound, ymax = Upper_Bound), fill = "gray", alpha = 0.5) +
  labs(title = "Gaussian Process Regression Predictions with Uncertainty",
       x = "Actual Critical Temperature",
       y = "Predicted Critical Temperature") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5)) 


#Error bar
ggplot(GPR_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_errorbar(aes(ymin = Lower_Bound, ymax = Upper_Bound), color = "red", alpha = 0.5) +
  labs(title = "Gaussian Process Predictions with Uncertainty", x = "Sample Index", y = "Predicted Critical Temperature")  +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5)) 


#Plot of GPR Model Residuals/Error Distribution
ggplot(GPR_df, aes(x = Actual, y = Actual - Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residual Plot (Actual - Predicted)",
       x = "Actual Critical Temperature",
       y = "Residuals") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))


#KPCA
# Kernel PCA using the Radial Basis Function (RBF)
kpca_result = kpca(~., data = as.data.frame(X_train_scaled), kernel = "rbfdot", kpar= list(sigma=0.1), features = 20)

# Standard deviations of principal components
pc_sd = apply(rotated(kpca_result), 2, sd)

# Plot of each component importance 
ggplot(data.frame(PC = 1:length(pc_sd), Variance = pc_sd^2), aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  labs(title = "Kernel PCA - Importance of Components", x = "Principal Component", y = "Variance Explained") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))


# Plot of KPCA-transformed data
ggplot(df_kpca, aes(x = V1, y = V2, color = critical_temp)) +
  geom_point(alpha = 0.6) +
  labs(title = "Kernel PCA Projection (Non-Linear Components)",
       x = "First Principal Component",
       y = "Second Principal Component") +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))


# Transforming training and test data
X_train_kpca = predict(kpca_result, as.data.frame(X_train_scaled))
X_test_kpca = predict(kpca_result, as.data.frame(X_test_scaled))


# Scaling X_kpca using Min-Max normalization
X_train_kpcanorm = normalize(as.data.frame(X_train_kpca), method = "range", range = c(0, 1))

# Normalizing X_test_kpca using the same min and max as X_train_kpca
X_test_kpcanorm = as.data.frame(lapply(seq_len(ncol(X_test_kpca)), function(i) {
  (X_test_kpca[, i] - min(X_train_kpca[, i])) / (max(X_train_kpca[, i]) - min(X_train_kpca[, i]))
}))

colnames(X_test_kpcanorm) = colnames(X_train_kpca)


# Training GPR on transformed KPCA training set
gprkpca_model = gausspr(x = X_train_kpcanorm, y = y_train, kernel = "rbfdot", kpar= list(sigma=0.1), variance.model = TRUE)

# Predictions on the transformed KPCA test set
gprkpca_pred = predict(gprkpca_model, X_test_kpcanorm)

# GPRKPCA Model Evaluation
gprkpca_mse = mean((y_test - gprkpca_pred)^2)
gprkpca_rmse = sqrt(gprkpca_mse)
gprkpca_mae = mean(abs(y_test - gprkpca_pred))
gprkpca_r2 = cor(y_test, gprkpca_pred)^2

print(c(GPRKPCA_MSE = gprkpca_mse, GPRKPCA_RMSE = gprkpca_rmse, GPRKPCA_MAE = gprkpca_mae, GPRKPCA_R2 = gprkpca_r2))

# Approximating gpr_pred for variance
gprkpca_var = kernlab::predict(gprkpca_model, X_test_kpcanorm, type = "probabilities")  
gprkpca_sd = sqrt(gprkpca_var)

# Data frame of GPRKPCA results for plotting
GPRKPCA_df = data.frame(Actual = y_test, Predicted = gprkpca_pred, 
                    Upper_Bound = gprkpca_pred + 1.96 * gprkpca_sd, Lower_Bound = gprkpca_pred - 1.96 * gprkpca_sd)


# Plot of predictions with uncertainty bands
ggplot(GPRKPCA_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_ribbon(aes(ymin = Lower_Bound, ymax = Upper_Bound), fill = "gray", alpha = 0.5) +
  labs(title = "Gaussian Process Regression (with KPCA) Predictions with Uncertainty",
       x = "Actual Critical Temperature",
       y = "Predicted Critical Temperature") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5)) 


#GPRKPCA Model Residuals/Error Distribution
ggplot(GPRKPCA_df, aes(x = Actual, y = Actual - Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "KPCA Residual Plot (Actual - Predicted)",
       x = "Actual Critical Temperature",
       y = "Residuals") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))



#PCA
pca_train = prcomp(X_train_scaled, scale. = TRUE)

# Proportion of variance explained by each principal component
summary(pca_train)

# Plot of cumulative variance explained
cumulative_variance = cumsum(pca_train$sdev^2 / sum(pca_train$sdev^2))
ggplot(data.frame(PC = 1:length(cumulative_variance), CumulativeVariance = cumulative_variance), aes(x = PC, y = CumulativeVariance)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Cumulative Variance Explained by Principal Components", x = "Principal Components", y = "Cumulative Variance") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))

# 95% components
n_components = which(cumulative_variance >= 0.95)[1]
cat("Number of components to retain:", n_components, "\n")

# Transforming the training and test data using the selected number of components
X_train_pca = as.data.frame(pca_train$x[, 1:n_components])
X_test_pca = as.data.frame(predict(pca_train, X_test_scaled)[, 1:n_components])


# Plot of each component importance
ggplot(data.frame(PC = 1:length(pca_train$sdev), Variance = pca_train$sdev^2), aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  labs(title = "PCA - Importance of Components", x = "Principal Component", y = "Variance Explained") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))

dim(X_train_pcanorm)
dim(X_test_pcanorm)

# Scaling X_pca using Min-Max normalization
X_train_pcanorm = normalize(as.data.frame(X_train_pca), method = "range", range = c(0, 1))

# Normalizing X_test_pca using the same min and max as X_train_pca
X_test_pcanorm = as.data.frame(lapply(seq_len(ncol(X_test_pca)), function(i) {
  (X_test_pca[, i] - min(X_train_pca[, i])) / (max(X_train_pca[, i]) - min(X_train_pca[, i]))
}))

colnames(X_test_pcanorm) = colnames(X_train_pca)


#GPR with PCA Model

gprpca_model = gausspr(x = X_train_pcanorm, y = y_train, kernel = "rbfdot", kpar= list(sigma=0.1), variance.model = TRUE)


# GPRPCA predictions
gprpca_pred = predict(gprpca_model, X_test_pcanorm)


# Evaluating the GPRPCA Model
gprpca_mse = mean((y_test - gprpca_pred)^2)
gprpca_rmse = sqrt(gprpca_mse)
gprpca_mae = mean(abs(y_test - gprpca_pred))
gprpca_r2 = cor(y_test, gprpca_pred)^2

print(c(GPRPCA_MSE = gprpca_mse, GPRPCA_RMSE = gprpca_rmse, GPRPCA_MAE = gprpca_mae, GPRPCA_R2 = gprpca_r2))


#PCA Uncertainties
# Approximating gprpca_pred for the variance
gprpca_var = kernlab::predict(gprpca_model, X_test_pcanorm, type = "probabilities")  

gprpca_sd = sqrt(gprpca_var)


# Data frame of GPR results for plotting
GPRPCA_df = data.frame(Actual = y_test, Predicted = gprpca_pred, 
                        Upper_Bound = gprpca_pred + 1.96 * gprpca_sd, Lower_Bound = gprpca_pred - 1.96 * gprpca_sd)



# Plot of GPRCA predictions with uncertainty bands
ggplot(GPRPCA_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_ribbon(aes(ymin = Lower_Bound, ymax = Upper_Bound), fill = "gray", alpha = 0.5) +
  labs(title = "Gaussian Process Regression (with PCA) Predictions with Uncertainty",
       x = "Actual Critical Temperature",
       y = "Predicted Critical Temperature") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5)) 


#Plot of GPRPCA Model Residuals/Error Distribution
ggplot(GPRPCA_df, aes(x = Actual, y = Actual - Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "PCA Residual Plot (Actual - Predicted)",
       x = "Actual Critical Temperature",
       y = "Residuals") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))



#ML MODELS for comparison
# Definining a function to select models
#List of ML Models
models = list(
  "Linear Regression" = lm(y_train ~ ., data = X_train_scaled),
  "Random Forest" = randomForest(x = X_train_scaled, y = y_train, ntree = 100, mtry = sqrt(ncol(X_train))),
  "Support Vector Regression" = svm(x = as.matrix(X_train_scaled), y = y_train, kernel = "radial")
  )

# Models' Evaluation Function
evaluate_model = function(model, test_data, y_test) {
  predictions = predict(model, test_data)
  mse = mean((y_test - predictions)^2)
  rmse = sqrt(mse)
  mae = mean(abs(y_test - predictions))
  r2 = cor(y_test, predictions)^2
  return(c(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2))
}

# Store the ML results
ML_results = lapply(models, evaluate_model, test_data = X_test_scaled, y_test = y_test)
ML_df = as.data.frame(do.call(rbind, ML_results))
ML_df = rownames_to_column(ML_df, var="Model")
print(ML_df)


#Combining all the GPR Models evaluations

GPR_results = list(c("GPR", gpr_mse, gpr_rmse, gpr_mae, gpr_r2), 
                   c("GPR_PCA",gprpca_mse, gprpca_rmse, gprpca_mae, gprpca_r2), 
                   c("GPR_KPCA",gprkpca_mse, gprkpca_rmse, gprkpca_mae, gprkpca_r2))

#GPR results to dataframe
new_df = as.data.frame(do.call(rbind, GPR_results))

#Uniformirm headings
colnames(new_df) = colnames((ML_df))

#combine all the results in a dataframes
results_df = rbind(new_df, ML_df)

print(results_df)

results_df$Model = as.factor(results_df$Model)

#Evaluation columns to numeric data type
numeric_cols = c("MSE", "RMSE", "MAE", "R2")
results_df[numeric_cols] = lapply(results_df[numeric_cols], as.numeric)

# Round all numeric columns to 2 decimal places
results_df[numeric_cols] = lapply(results_df[numeric_cols], function(column) {
  round(column, 2)})

print(results_df)


#Bar plot of the Models R² performance
ggplot(results_df, aes(x = Model, y = R2, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Performance Comparison (R² Score)", y = "R² Score", fill="Model") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))


#Bar plot of the MOdels RSME performance
ggplot(results_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Performance Comparison (RMSE)", y = "RMSE", fill="Model") +
  theme_minimal() +  theme(plot.title = element_text(hjust = 0.5))









# ----- NOT ENOUGH MEMMORY TO TRAIN WITH THE THE TWO LIBRARIES BELOW DESPITE THIER UNCERTAINTY CAPABILITIES ----

library(DiceKriging)

# Train GPR Model
gpr_model2 = km(formula = ~., design = X_train_scaled, response = y_train, covtype = "gauss",
                nugget = 1e-6, control = list(trace = FALSE)) 

# Make predictions
gpr_pred2 = predict(gpr_model2, newdata = X_test_scaled, type = "UK")



library(GPfit)    # For Gaussian Process modeling with uncertainty

# Train GP model
gp_model3 = GP_fit(X = X_train_scaled, Y = y_train, corr = list(type = "exponential", power = 2))

# Make predictions with uncertainty
gp_pred3 <- predict(gp_model2, xnew = X_test_scaled)


