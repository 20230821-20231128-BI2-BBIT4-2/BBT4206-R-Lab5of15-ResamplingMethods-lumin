Business Intelligence Project
================
<Specify your name here>
<Specify the date when you submitted the lab>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [STEP 1. Install and Load the Required
  Packages](#step-1-install-and-load-the-required-packages)
  - [caret](#caret)
  - [klaR](#klar)
  - [e1071](#e1071)
  - [readr](#readr)
  - [LiblineaR](#liblinear--)
  - [naivebayes](#naivebayes--)
  - [mlbench](#mlbench--)
- [CASE 1: Splitting dataset into train and test sets](#case-1-splitting-dataset-into-train-and-test-sets)    
  - [STEP 2. Training the model, Testing the model and Viewing the Results](#step-2-training-the-model-testing-the-model-and viewing-the-results)
     -[First Option: using e1071 package](#first-option-using-e1071-package)
     - [Second Option: using caret package](#second-option-using-caret-package)
     - [Third Option: using klar package](#third-option-using-klar-package)
- [CASE 2: Bootstrapping](#bootstrapping)
- [CASE 3. CV, Repeated CV, and LOOCV](#case-3-cv-repeated-cv-and-loocv)
  - [Option 1. Regression: Linear Model (10-fold Cross
    Validation)](#option-1-regression-linear-model-10-fold-cross-validation)
  - [Option 2. Classification: LDA with k-fold Cross Validation](#option-2-classification-lda-with-k-fold-cross-validation)
  - [Option 3. Classification: Naive Bayes with Repeated k-fold Cross
    Validation](#option-3-classification-naive-bayes-with-repeated-k-fold-cross-validation)
  - [Option 4. Classification: SVM with Repeated k-fold Cross Validation](#classification-svm-with-repeated-k-fold-cross-validation)
  - [Option 5. Classification: Naive Bayes with Leave One Out Cross
    Validation](#option-5-classification-naive-bayes-with-leave-one-out-cross-validation)

# Student Details

|                                              |                                              |
|----------------------------------------------|----------------------------------------------|
| **Student ID Number**                        | 112827,132234,134265                         |
| **Student Name**                             | Kenneth Mungai,Kelly Noella, Emmanuel Kiptoo |
| **BBIT 4.2 Group**                           | A,B                                          |
| **BI Project Group Name/ID (if applicable)** | Lumin                                        |

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

# STEP 1. Install and Load the Required Packages

\`\`\`r {step-one chunck} if (require(“languageserver”)) {
require(“languageserver”) } else { install.packages(“languageserver”,
dependencies = TRUE, repos = “<https://cloud.r-project.org>”) }

### caret —-

if (require(“caret”)) { require(“caret”) } else {
install.packages(“caret”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

### klaR —-

if (require(“klaR”)) { require(“klaR”) } else { install.packages(“klaR”,
dependencies = TRUE, repos = “<https://cloud.r-project.org>”) }

### e1071 —-

if (require(“e1071”)) { require(“e1071”) } else {
install.packages(“e1071”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

### readr —-

if (require(“readr”)) { require(“readr”) } else {
install.packages(“readr”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

### LiblineaR —-

if (require(“LiblineaR”)) { require(“LiblineaR”) } else {
install.packages(“LiblineaR”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

### naivebayes —-

if (require(“naivebayes”)) { require(“naivebayes”) } else {
install.packages(“naivebayes”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

### mlbench —-

if (require(“mlbench”)) { require(“mlbench”) } else {
install.packages(“mlbench”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }


    # STEP 2. Load the dataset
    ```r{step-two chunck}
    data(PimaIndiansDiabetes)
    summary(PimaIndiansDiabetes)

    str(PimaIndiansDiabetes)

\#CASE 1: Splitting dataset into train and test sets \## STEP 1.
Splitting dataset
`r{case 1.1 chunck} train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,                                    p = 0.75,                                    list = FALSE) pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ] pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]`

## STEP 2. Training the model, Testing the model and Viewing the Results

\`\`\`r{case 1.2} \# First Option: using e1071 package \## training
PimaIndiansDiabetes_model_nb_e1071 \<- \# nolint
e1071::naiveBayes(diabetes ~ pregnant + glucose + pressure + triceps +
insulin + mass + pedigree + age, data = pima_indians_diabetes_train) \##
testing predictions_nb_e1071 \<-
predict(PimaIndiansDiabetes_model_nb_e1071,
pima_indians_diabetes_test\[, c(“pregnant”, “glucose”, “pressure”,
“triceps”, “insulin”, “mass”,“pedigree”, “age”)\]) \## results  
print(predictions_nb_e1071) caret::confusionMatrix(predictions_nb_e1071,
pima_indians_diabetes_test\[, c(“pregnant”, “glucose”, “pressure”,
“triceps”, “insulin”, “mass”,“pedigree”, “age”, “diabetes”)\]\$diabetes)

plot(table(predictions_nb_e1071, pima_indians_diabetes_test\[,
c(“pregnant”, “glucose”, “pressure”, “triceps”, “insulin”,
“mass”,“pedigree”, “age”, “diabetes”)\]\$diabetes))

# Second Option: using caret package

## training

PimaIndiansDiabetes_model_nb_caret \<- \# nolint caret::train(diabetes ~
., data = pima_indians_diabetes_train\[, c(“pregnant”, “age”, “glucose”,
“pressure”, “triceps”, “insulin”, “mass”, “pedigree”, “diabetes” )\],
method = “naive_bayes”) \## testing  
predictions_nb_caret \<- predict(PimaIndiansDiabetes_model_nb_caret,
pima_indians_diabetes_test\[, c(“pregnant”, “glucose”, “pressure”,
“triceps”, “insulin”, “mass”,“pedigree”, “age”)\])

## results

print(PimaIndiansDiabetes_model_nb_caret)
caret::confusionMatrix(predictions_nb_caret,
pima_indians_diabetes_test\[, c(“pregnant”, “glucose”, “pressure”,
“triceps”, “insulin”, “mass”,“pedigree”, “age”, “diabetes”)\]\$diabetes)

plot(table(predictions_nb_caret, pima_indians_diabetes_test\[,
c(“pregnant”, “glucose”, “pressure”, “triceps”, “insulin”,
“mass”,“pedigree”, “age”, “diabetes”)\]\$diabetes))

# Third Option: using klar package

PimaIndiansDiabetes_model_nb_klaR \<- \# nolint
klaR::NaiveBayes(`diabetes` ~ ., data = pima_indians_diabetes_train)

    # CASE 2. Bootstrapping
    ```r{case-two chunck}
    #splitting
    train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                       p = 0.75,
                                       list = FALSE)
    pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
    pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

    #training
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
            
    #testing
    predictions_logit <- predict(PimaIndiansDiabetes_model_logit,
                              pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                             "pressure", "triceps",
                                                             "insulin",
                                                             "mass","pedigree", 
                                                             "age",
                                                             "diabetes")])

    #results   ----
    print(PimaIndiansDiabetes_model_logit)
    print(predictions_logit)

# CASE 3. CV, Repeated CV, and LOOCV

## Option 1. Regression: Linear Model (10-fold Cross Validation)

\`\`\`r{case 3.1 chunck} \#training train_control \<-
trainControl(method = “cv”, number = 10)

PimaIndiansDiabetes_model_lm \<- \# nolint train(`diabetes` ~
`pregnant` + `glucose` + `pressure` + `triceps` + `insulin` + `mass` +
`pedigree` + `age`, data = pima_indians_diabetes_train, trControl =
train_control, na.action = na.omit, method = “glm”, metric = “Accuracy”)

\#testing predictions_lm \<- predict(PimaIndiansDiabetes_model_lm,
pima_indians_diabetes_test\[,c(“pregnant”, “glucose”, “pressure”,
“triceps”, “insulin”, “mass”,“pedigree”, “age”, “diabetes”)\])

\#viewing results (RMSE) —- print(PimaIndiansDiabetes_model_lm)
print(predictions_lm)


    ## Option 2. Classification: LDA with k-fold Cross Validation (LDA classifier based on a 5-fold CV)  

    ```r{case 3.2 chucnk}
    #training  ----
    PimaIndiansDiabetes_model_lda <-
      caret::train(`diabetes` ~ ., data = pima_indians_diabetes_train,
                   trControl = train_control, na.action = na.omit, method = "lda2",
                   metric = "Accuracy")

    # test the trained LDA model   ----
    predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                               pima_indians_diabetes_test[, c("pregnant", "glucose",
                                                              "pressure", "triceps",
                                                              "insulin",
                                                              "mass","pedigree", 
                                                              "age",
                                                              "diabetes")])

    # view the summary of the model and view the confusion matrix ----
    print(PimaIndiansDiabetes_model_lda)
    caret::confusionMatrix(predictions_lda, pima_indians_diabetes_test$diabetes)

## Option 3. Classification: Naive Bayes with Repeated k-fold Cross Validation —-

\`\``r{case 3.3 chunck} # train an e1071::naive Bayes classifier based on the diabetes variable ---- PimaIndiansDiabetes_model_nb <-   e1071::naiveBayes(`diabetes\`
~ ., data = pima_indians_diabetes_train)

# test the trained naive Bayes classifier —-

predictions_nb_e1071 \<- predict(PimaIndiansDiabetes_model_nb,
pima_indians_diabetes_test\[, 1:9\])

# View a summary of the naive Bayes model and the confusion matrix —-

print(PimaIndiansDiabetes_model_nb)
caret::confusionMatrix(predictions_nb_e1071,
pima_indians_diabetes_test\$diabetes)


    ## Option 4. Classification: SVM with Repeated k-fold Cross Validation ----

    ```r{case 3.4 chunck}
    # SVM Classifier using 5-fold cross validation with 3 reps ----
    # training

    train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

    PimaIndiansDiabetes_model_svm <-
      caret::train(`diabetes` ~ ., data = pima_indians_diabetes_train,
                   trControl = train_control, na.action = na.omit,
                   method = "svmLinearWeights2", metric = "Accuracy")

    # test the trained SVM model ----
    predictions_svm <- predict(PimaIndiansDiabetes_model_svm, pima_indians_diabetes_test[, 1:9])

    # View a summary of the model and view the confusion matrix ----
    print(PimaIndiansDiabetes_model_svm)
    caret::confusionMatrix(predictions_svm, pima_indians_diabetes_test$diabetes)

## Option 5. Classification: Naive Bayes with Leave One Out Cross Validation —-

\`\`\`r{case 3.5 chunck} \# Train a Naive Bayes classifier based on an
LOOCV —- train_control \<- trainControl(method = “LOOCV”)

PimaIndiansDiabetes_nb_loocv \<- caret::train(`diabetes` ~ ., data =
pima_indians_diabetes_train, trControl = train_control, na.action =
na.omit, method = “naive_bayes”, metric = “Accuracy”)

# Test the trained model using the testing dataset —-

predictions_nb_loocv \<- predict(PimaIndiansDiabetes_nb_loocv,
pima_indians_diabetes_test\[, 1:9\])

# View the confusion matrix —-

print(PimaIndiansDiabetes_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv,
pima_indians_diabetes_test\$diabetes)

\`\`\`
