### LOAD REQUIRED LIBRARIES
library(class)
library(e1071)
library(caret)
library(rpart.plot)
library(Boruta)
library(ggplot2)
library(ranger)
library(dplyr)
library(corrplot)
library(pROC)
library(reshape2)
library(shiny)
library(xgboost)


### DATASET DESCRIPTION
# Load bank data
bankdata <- read.table('bank-full.csv', sep = ',', header = TRUE)
# Load dataset description
datadesc <- read.table('DatasetTable.csv', sep = ',', header = TRUE)
# Modify column names of dataset description
colnames(datadesc) <- c('ATTRIBUTE NAME', 'ATTRIBUTE DESCRIPTION (CATEGORY VALUES)', 'DATA TYPE')

### DATASET PRE-PROCESSING
# Check for missing values
sapply(bankdata, function(x) sum(is.na(x)))
# Check if there are duplicated values
sum(duplicated(bankdata))

bankdata$y <- ifelse(bankdata$y == "yes", 1, 0)

### EXPLORATORY DATA ANALYSIS
# Plot histograms of numeric attributes
par(mfrow = c(3,2))
hist(bankdata$age)
hist(bankdata$duration)
hist(bankdata$balance)
hist(bankdata$campaign)
hist(bankdata$previous)
hist(bankdata$pdays)

# Output variable bar plot
ggplot(data = bankdata, aes(x = y, fill = y)) + geom_bar(stat = "count")

# Correlation matrix for numeric variables
corr_matrix <- cor(bankdata[sapply(bankdata, is.numeric)])
corrplot(corr_matrix, method = "color", addCoef.col = "black")

numeric_vars <- sapply(bankdata, is.numeric)

# Bar plots for categorical variables
categorical_vars <- names(bankdata)[!numeric_vars]
for (cat_var in categorical_vars) {
  # Calculate percentages
  data_for_plot <- bankdata%>%
    group_by_at(vars(cat_var, "y")) %>%
    summarise(count = n()) %>%
    mutate(percentage = count / sum(count) * 100)
  
  # Create the plot
  p <- ggplot(data = data_for_plot, aes_string(x = cat_var, y = "count", fill = "y")) + 
    geom_bar(stat = "identity") + 
    geom_text(aes(label = sprintf("%.1f%%", percentage), y = count/2), position = position_dodge(0.9), size = 3) + 
    theme_minimal()
  
  print(p)
}


### MODEL BUILDING AND EVALUATION
# Set seed for reproducibility
# Split the dataset into training (60%), validation (20%), and test sets (20%)
set.seed(0)
trainIndex <- createDataPartition(bankdata$y, p = .6, list = FALSE)
trainData <- bankdata[ trainIndex,]
tempData <- bankdata[-trainIndex,]
validIndex <- createDataPartition(tempData$y, p = .5, list = FALSE)
validData <- tempData[ validIndex,]
testData <- tempData[-validIndex,]


## Logistic Regression
logistic_model <- glm(y ~ ., family = "binomial", data = trainData)
pred_logistic_valid <- predict(logistic_model, validData, type = "response")
# Convert predicted values to factor
pred_factor <- as.factor(ifelse(pred_logistic_valid > 0.5, 1, 0))

# Convert actual values to factor (if not already)
validData$y <- as.factor(validData$y)

# Make sure both factors have the same levels
levels(pred_factor) <- union(levels(pred_factor), levels(validData$y))
levels(validData$y) <- union(levels(pred_factor), levels(validData$y))

# Logistic Regression Confusion Matrix
cm_logistic <- confusionMatrix(pred_factor, validData$y)


## Decision Tree
trainData$y <- as.factor(trainData$y)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
dtree_fit <- train(y ~., data = trainData, method = "rpart",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)
pred_dt_valid <- predict(dtree_fit, validData, type = "raw")

# Ensure both are factors
pred_dt_valid <- as.factor(pred_dt_valid)
validData$y <- as.factor(validData$y)

# Make sure both factors have the same levels
levels(pred_dt_valid) <- union(levels(pred_dt_valid), levels(validData$y))
levels(validData$y) <- union(levels(pred_dt_valid), levels(validData$y))

# Decision Tree Confusion Matrix
cm_dt <- confusionMatrix(pred_dt_valid, validData$y)

## RandomForest
# Model Results
trainData$y <- as.factor(trainData$y)
rf_fit <-ranger(y ~ .,  
                      data = trainData,  
                      importance = "impurity") 
rf_pred=predict(rf_fit,validData)

rf_pred_class <- as.factor(rf_pred$predictions)

# Ensure both are factors
validData$y <- as.factor(validData$y)

# Make sure both factors have the same levels
levels(rf_pred_class) <- union(levels(rf_pred_class), levels(validData$y))
levels(validData$y) <- union(levels(rf_pred_class), levels(validData$y))

# Random Forest Confusion Matrix
cm_rf <- confusionMatrix(rf_pred_class, validData$y)

# Feature importance visualization
var_importance <- data.frame(Variable = names(rf_fit$variable.importance),
                             Importance = rf_fit$variable.importance)

ggplot(var_importance, aes(x = reorder(Variable, -Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance from Random Forest",
       x = "Variable",
       y = "Importance") +
  theme_minimal()

# Initialize the comparison table
comparison_table <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  BalancedAccuracy = numeric(),
  stringsAsFactors = FALSE
)

# Function to compute metrics from a confusion matrix and add to the table
add_to_table <- function(cm, model_name) {
  TN <- cm$table[1,1]
  FP <- cm$table[1,2]
  FN <- cm$table[2,1]
  TP <- cm$table[2,2]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  balanced_accuracy <- (sensitivity + specificity) / 2
  
  # Add results to the comparison table
  return(data.frame(
    Model = model_name,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    BalancedAccuracy = balanced_accuracy
  ))
}

# Populate the comparison table
comparison_table <- rbind(comparison_table, add_to_table(cm_logistic, "Logistic Regression"))
comparison_table <- rbind(comparison_table, add_to_table(cm_dt, "Decision Tree"))
comparison_table <- rbind(comparison_table, add_to_table(cm_rf, "Random Forest"))


# Display the comparison table
print(comparison_table)

# Melt the data
melted_data <- melt(comparison_table, id.vars = "Model")

# Bar plot
ggplot(melted_data, aes(x = Model, y = value, fill = variable)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  labs(y = "Value", x = "Model", title = "Model Comparison", fill = "Metric") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))





