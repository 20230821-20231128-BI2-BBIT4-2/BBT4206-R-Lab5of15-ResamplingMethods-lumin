# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)



if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## Resampling methods include: ----
### Splitting the dataset into train and test sets ----
### Bootstrapping (sampling with replacement) ----
### Basic k-fold cross validation ----
### Repeated cross validation ----
### Leave One Out Cross-Validation (LOOCV) ----

## STEP 1. Install and Load the Required Packages ----

### caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

### mlbench ----
install.packages("mlbench")
library(mlbench)


## STEP 2. Load the dataset ----
data(PimaIndiansDiabetes)
summary(PimaIndiansDiabetes)

# The str() function is used to compactly display the structure (variables
# and data types) of the dataset
str(PimaIndiansDiabetes)

# Splitting the dataset into train and test sets   ----

### 1. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# 75% to train and 25% to test the model

train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### 2. Train the model ----
#### 2.a OPTION 1: naiveBayes() function in the e1071 package ----

PimaIndiansDiabetes_model_nb_e1071 <- # nolint
  e1071::naiveBayes(diabetes ~ pregnant + glucose + 
                      pressure + triceps +
                      insulin +
                      mass + pedigree +
                      age,
                    data = pima_indians_diabetes_train)
 
#### 2.b. OPTION 2: naiveBayes() function in the caret package ----
PimaIndiansDiabetes_model_nb_caret <- # nolint
  caret::train(diabetes ~ ., data =
                 pima_indians_diabetes_train[, c("pregnant", "age", "glucose",
                                             "pressure", "triceps", "insulin",
                                             "mass",
                                             "pedigree",
                                             "diabetes"
                                             )],
               method = "naive_bayes")

#### 2.c OPTION 3: "NaiveBayes()" function in the "klaR" package ----
PimaIndiansDiabetes_model_nb_klaR <- # nolint
  klaR::NaiveBayes(`diabetes` ~ .,
                   data = pima_indians_diabetes_train)


### 3. Test the model using the testing dataset ----
#### 3.a Test the e1071 Naive Bayes model ----

predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb_e1071,
          pima_indians_diabetes_test[, c("pregnant", "glucose",
                                     "pressure", "triceps", "insulin",
                                     "mass","pedigree", "age")])

#### 3.b Test the caret Naive Bayes model ----

predictions_nb_caret <-
  predict(PimaIndiansDiabetes_model_nb_caret,
          pima_indians_diabetes_test[, c("pregnant", "glucose",
                                         "pressure", "triceps", "insulin",
                                         "mass","pedigree", "age")])

### 4. Results ----
#### 4.a e1071 Naive Bayes model ----
print(predictions_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                      "pressure", "triceps",
                                                      "insulin",
                                                      "mass","pedigree", 
                                                      "age",
                                                      "diabetes")]$diabetes)

plot(table(predictions_nb_e1071,
           pima_indians_diabetes_test[, c("pregnant", "glucose",
                                          "pressure", "triceps",
                                          "insulin",
                                          "mass","pedigree", 
                                          "age",
                                          "diabetes")]$diabetes))

#### 4. b caret Naive Bayes model ----
print(PimaIndiansDiabetes_model_nb_caret)
caret::confusionMatrix(predictions_nb_caret,
                       pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                      "pressure", "triceps",
                                                      "insulin",
                                                      "mass","pedigree", 
                                                      "age",
                                                      "diabetes")]$diabetes)

plot(table(predictions_nb_caret,
           pima_indians_diabetes_test[, c("pregnant", "glucose",
                                          "pressure", "triceps",
                                          "insulin",
                                          "mass","pedigree", 
                                          "age",
                                          "diabetes")]$diabetes))
# Bootstrapping  ----
### 1. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# 75% to train and 25% to test the model

train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### 2. Train a  classification model ----
#### 2.a Train control  ----
train_control <- trainControl(method = "cv", number = 10)

PimaIndiansDiabetes_model_logit <- # nolint
  train(`diabetes` ~
                 `pregnant` + `glucose` +
                 `pressure` + `triceps` +
                 `insulin` + `mass` +
                 `pedigree` +
                 `age`,
               data = pima_indians_diabetes_train,
               trControl = train_control,
               na.action = na.omit, method = "glm", family ="binomial", 
        metric = "Accuracy")

### 3. Test the trained linear regression model   ----
predictions_logit <- predict(PimaIndiansDiabetes_model_logit,
                          pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                         "pressure", "triceps",
                                                         "insulin",
                                                         "mass","pedigree", 
                                                         "age",
                                                         "diabetes")])

### 4. Results   ----
print(PimaIndiansDiabetes_model_logit)
print(predictions_logit)


# CV, Repeated CV, and LOOCV   ----
### 1. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# 75% to train and 25% to test the model

train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### 2. Regression: Linear model      ----
#### 2.a  10-fold cross validation  ----
train_control <- trainControl(method = "cv", number = 10)

PimaIndiansDiabetes_model_lm <- # nolint
  train(`diabetes` ~
          `pregnant` + `glucose` +
          `pressure` + `triceps` +
          `insulin` + `mass` +
          `pedigree` +
          `age`,
        data = pima_indians_diabetes_train,
        trControl = train_control,
        na.action = na.omit, method = "glm", 
        metric = "Accuracy")

#### 2.b Test the trained model   ----
predictions_lm <- predict(PimaIndiansDiabetes_model_lm, pima_indians_diabetes_test[,c("pregnant", "glucose",
                                                                                      "pressure", "triceps",
                                                                                      "insulin",
                                                                                      "mass","pedigree", 
                                                                                      "age",
                                                                                      "diabetes")])

### 2.c. View the RMSE and the predicted values ====
print(PimaIndiansDiabetes_model_lm)
print(predictions_lm)

### 3. Classification: LDA with k-fold Cross Validation   ----
#### 3.a LDA classifier based on a 5-fold CV    ----
PimaIndiansDiabetes_model_lda <-
  caret::train(`diabetes` ~ ., data = pima_indians_diabetes_train,
               trControl = train_control, na.action = na.omit, method = "lda2",
               metric = "Accuracy")

#### 3.b Test the trained LDA model   ----
predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                           pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                          "pressure", "triceps",
                                                          "insulin",
                                                          "mass","pedigree", 
                                                          "age",
                                                          "diabetes")])

#### 3.c. View the summary of the model and view the confusion matrix ----
print(PimaIndiansDiabetes_model_lda)
caret::confusionMatrix(predictions_lda, pima_indians_diabetes_test$diabetes)


### 4. Classification: Naive Bayes with Repeated k-fold Cross Validation ----
#### 4.a. Train an e1071::naive Bayes classifier based on the diabetes variable ----
PimaIndiansDiabetes_model_nb <-
  e1071::naiveBayes(`diabetes` ~ ., data = pima_indians_diabetes_train)

#### 4.b. Test the trained naive Bayes classifier   ----
predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb, pima_indians_diabetes_test[, 1:9])

#### 4.c. View a summary of the naive Bayes model and the confusion matrix ----
print(PimaIndiansDiabetes_model_nb)
caret::confusionMatrix(predictions_nb_e1071, pima_indians_diabetes_test$diabetes)

### 5. Classification: SVM with Repeated k-fold Cross Validation ----
#### 5.a. SVM Classifier using 5-fold cross validation with 3 reps ----


train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <-
  caret::train(`diabetes` ~ ., data = pima_indians_diabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")

#### 5.b. Test the trained SVM model ----
predictions_svm <- predict(PimaIndiansDiabetes_model_svm, pima_indians_diabetes_test[, 1:9])

#### 5.c. View a summary of the model and view the confusion matrix ----
print(PimaIndiansDiabetes_model_svm)
caret::confusionMatrix(predictions_svm, pima_indians_diabetes_test$diabetes)


### 6. Classification: Naive Bayes with Leave One Out Cross Validation ----


#### 6.a. Train a Naive Bayes classifier based on an LOOCV ----
train_control <- trainControl(method = "LOOCV")

PimaIndiansDiabetes_nb_loocv <-
  caret::train(`diabetes` ~ ., data = pima_indians_diabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")

#### 6.b. Test the trained model using the testing dataset ====
predictions_nb_loocv <-
  predict(PimaIndiansDiabetes_nb_loocv, pima_indians_diabetes_test[, 1:9])

#### 6.c. View the confusion matrix ====
print(PimaIndiansDiabetes_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv, pima_indians_diabetes_test$diabetes)
