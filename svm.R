# Data wrangling 
library(tidyverse)
library(janitor)
library(modelr)
library(rsample)
library(knitr)
library(caret)


# Fitting and examining SVM
library(e1071)

red_wine <- read.csv("winequality-red.csv", 
                     sep=";", header=TRUE) %>%
  as_tibble() %>%
  clean_names() %>% 
  mutate_if(is_character, factor) %>%
  mutate(quality = factor(quality))

## 75% of the sample size
## set the seed to make your partition reproducible
set.seed(100) # setting seed
red_wine$id <- 1:nrow(red_wine)
train <- red_wine %>% dplyr::sample_frac(.75)
test  <- dplyr::anti_join(red_wine, train, by = 'id')


#Tuning CV
red_wine_linear_svm <- tibble(train = train %>% list()) %>%
  mutate(tune_svm_linear = map(.x = train, 
                               .f = function(x){ 
                                 return(tune(svm,
                                             quality ~ . - id,
                                             data = x, 
                                             kernel = "linear",
                                             ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))
                                 )
                               })
  )

(linear_parameters <- red_wine_linear_svm %>%
    pluck("tune_svm_linear", 1) %>%
    pluck("best.parameters"))

red_wine_linear_svm_model <- 
  tibble(train = train %>% list(),
         test = test %>% list()) %>%
  mutate( 
    model_fit = map(.x = train,
                    .f = function(x) svm(quality ~ . -id, 
                                         data = x, 
                                         cost = linear_parameters$cost, 
                                         kernel = "linear")), 
    test_pred = map2(model_fit, test, predict),
    # calculating confusion matrix
    confusion_matrix = map2(.x = test, .y = test_pred,
                            .f = function(x, y) caret::confusionMatrix(x$quality, y)))
red_wine_linear_svm_model %>% pluck("confusion_matrix", 1)

#Accuracy: 0.545; Test Error: 0.455

# part 2 ------------------------------------------------------------------

set.seed(100) # setting seed
red_wine_radial_svm <- tibble(train = train %>% list()) %>%
  mutate(tune_svm_radial = map(.x = train, 
                               .f = function(x){ 
                                 return(tune(svm, # svm
                                             quality ~ . -id, 
                                             data = x, 
                                             kernel = "radial", # radial
                                             ranges = list(cost = c(0.1, 0.5, 1, 5, 10, 50, 100, 200, 500)))
                                 )
                               })
  )

(linear_parameters <- red_wine_radial_svm %>%
    pluck("tune_svm_radial", 1) %>%
    pluck("best.parameters"))

#Im calculating the confusion matrix for selected model
red_wine_radial_svm_model <- 
  tibble(train = train %>% list(),
         test = test %>% list()) %>%
  mutate( 
    model_fit = map(.x = train,
                    .f = function(x) svm(quality ~ . -id, 
                                         data = x, 
                                         cost = linear_parameters$cost, 
                                         kernel = "radial")), 
    test_pred = map2(model_fit, test, predict),
    confusion_matrix = map2(.x = test, .y = test_pred,
                            .f = function(x, y) caret::confusionMatrix(x$quality, y)))

red_wine_radial_svm_model %>% pluck("confusion_matrix", 1)

#Accuracy : 0.6125; Test Error: 0.388  

# c -----------------------------------------------------------------------

set.seed(100) 
red_wine_polynomial_svm <- tibble(train = train %>% list()) %>%
  mutate(tune_svm_polynomial = map(.x = train, 
                                   .f = function(x){ 
                                     return(tune(svm, 
                                                 quality ~ . -id, 
                                                 data = x, 
                                                 kernel = "polynomial", 
                                                 ranges = list(cost = c(0.1, 1, 10, 50, 100, 500)))
                                     )
                                   })
  )

(linear_parameters <- red_wine_polynomial_svm %>%
    pluck("tune_svm_polynomial", 1) %>%
    pluck("best.parameters"))

red_wine_polynomial_svm_model <-  
  tibble(train = train %>% list(),
         test = test %>% list()) %>%
  mutate( 
    model_fit = map(.x = train,
                    .f = function(x) svm(quality ~ . -id, 
                                         data = x,
                                         cost = linear_parameters$cost, 
                                         kernel = "polynomial")), 
    test_pred = map2(model_fit, test, predict),
    confusion_matrix = map2(.x = test, .y = test_pred,
                            .f = function(x, y) caret::confusionMatrix(x$quality, y)))

red_wine_polynomial_svm_model %>% pluck("confusion_matrix", 1)

#Accuracy : 0.575; Error Rate: 0.425

# table -------------------------------------------------------------------

#Construct a table displaying the test error for these 3 candidate classifiers,
#the candidate models from L02 CART, and the multiple logistic regression fit in L02 CART.

tibble(
  model_type = c("Linear", "Random Forest", "XGBoost",
                 "Ridge Logistic", 
                 "SVM - linear", 
                 "SVM - radial", 
                 "SVM - polynomial"),
  tuning = c("fmla = quantity ~ . -id", "mtry = 5",
             "eta = 0.0217, nrounds = 220", "lambda = 0.026",
             "cost = 1, fmla = quality ~ . -id",
             "cost = 5, fmla = quality ~ . -id",
             "cost = 10, fmla = quality ~ . -id"),
  misclass_rate = c(0.407, 0.341, 0.381, 0.405, 0.455, 0.388, 0.425)
) %>%
  arrange(misclass_rate) %>%
  kable()