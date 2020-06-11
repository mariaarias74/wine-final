# Data wrangling 
library(tidyverse)
library(janitor)
library(modelr)
library(rsample)
library(knitr)
library(tidyselect)
library(ggplot2)

# For random forests
library(ranger)
library(vip)
library(pdp)

# For boosted models
library(xgboost)
library(glmnet)
library(glmnetUtils)

conflict_prefer("cv.glmnet", "glmnetUtils")
conflict_prefer("glmnet", "glmnetUtils")
conflict_prefer("partial", "pdp")

# Load data
red_wine <- read.csv("winequality-red.csv", 
                     sep=";", header=TRUE) %>%
  as_tibble() %>%
  clean_names()

## 75% of the sample size
## set the seed to make your partition reproducible
set.seed(4)
red_wine$id <- 1:nrow(red_wine)
train <- red_wine %>% dplyr::sample_frac(.75)
test  <- dplyr::anti_join(red_wine, train, by = 'id')


# Boosting Model ----------------------------------------------------------

#Helper Function to convert tibbles into Dmatrix
xgb_matrix <- function(dat, outcome, exclude_vars){
  dat_types <- dat %>% map_chr(class)
  outcome_type <- class(dat[[outcome]])
  if("chr" %in% dat_types){
    print("You must encode characters as factors.")
    return(NULL)
  } else {
    # If we're doing binary outcomes, they need to be 0-1
    if(outcome_type == "factor" & nlevels(dat[[outcome]]) == 2){
      tmp <- dat %>% select(outcome) %>% onehot::onehot() %>% predict(dat)  
      lab <- tmp[,1]
    } else {
      lab <- dat[[outcome]]
    }
    mat <- dat %>% dplyr::select(-outcome, -exclude_vars) %>% # encode on full boston df
      onehot::onehot() %>% # use onehot to encode variables
      predict(dat) # get OHE matrix
    return(xgb.DMatrix(data = mat, 
                       label = lab))
  }
}

#Helper function to get misclass rate of the test set
xg_error <- function(model, test_mat, metric = "mse"){
  # Get predictions and actual values
  preds = predict(model, test_mat)
  vals = getinfo(test_mat, "label")
  if(metric == "mse"){
    # Compute MSE
    err <- mean((preds - vals)^2)
  } else if(metric == "misclass") {
    # Otherwise, get the misclass rate
    err <- mean(preds != vals)
  }
  return(err)
}

#okay so now we're tuning
#My first parameter will be ETA or training rate
# Fitting xgboost
red_wine_xg_reg <- train %>%
  #using K-fold
  crossv_kfold(10, id = "folds") %>%
  crossing(eta = 10^seq(-10, -.1, length.out = 20)) %>%
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble),
    train_mat = map(train, xgb_matrix, outcome = "quality", exclude_vars = "id"), 
    test_mat = map(test, xgb_matrix, outcome = "quality", exclude_vars = "id"),
    xg_model = map2(.x = train_mat, .y = eta,
                    .f = function(x, y) xgb.train(params = list(eta = y, # set learning rate
                                                                depth = 10, # tree depth, can tune
                                                                objective = "reg:squarederror"), # binary class problem 
                                                  data = x, 
                                                  nrounds = 500,
                                                  silent = TRUE)),
    xg_train_mse = map2_dbl(xg_model, train_mat, xg_error, metric = "mse"), # train MSE
    xg_test_mse = map2_dbl(xg_model, test_mat, xg_error, metric = "mse") # test MSE
  )

#Im calculating the mean MSE across all the 10 folds
red_wine_xg_reg_meanmse <- red_wine_xg_reg %>% 
  group_by(eta) %>%
  summarise(mean_train_mse = mean(xg_train_mse),
            mean_test_mse = mean(xg_test_mse)) 

ggplot(red_wine_xg_reg_meanmse) + 
  geom_line(aes(eta, mean_train_mse, color = "Training Error")) +
  geom_line(aes(eta, mean_test_mse, color = "Test Error")) +
  scale_color_manual("", values = c("blue", "red")) + 
  labs(x = "Learning Rate", y = "MSE") +
  ylim(0, 15000)

#Saving the 3 best learning rate values
(candidate_etas <- red_wine_xg_reg_meanmse %>%
    arrange(mean_test_mse) %>%
    head(3) %>%
    select(eta))

#The optimal learning rate seems to be close to 0, so quite small. 
#There is a sharp decline of MSE from eta = 0 to the optimal eta which seems to be 0.0721 based on the results.

# Optimal number of trees ---------------------------------------------------------------

#Similar method as finding the optimal eta here by using 10-fold CV to fit and find the model with the lowest mean MSE. 
#Upon trial and error, I found 20~400 (20 nrounds values) a reasonable range. 
#Second tuning parameter- Optimal number of trees
red_wine_xg_reg_nrounds <- train %>%
  crossv_kfold(10, id = "folds") %>%
  crossing(nrounds = 10*seq(2, 40, length.out = 20)) %>% #testing 20 different nrounds
  mutate(
    train = map(train, as_tibble), 
    test = map(test, as_tibble), 
    train_mat = map(train, xgb_matrix, outcome = "quality", exclude_vars = "id"), 
    test_mat = map(test, xgb_matrix, outcome = "quality", exclude_vars = "id"),
    xg_model = map2(.x = train_mat, .y = nrounds,
                    .f = function(x, y) xgb.train(params = list(eta = 0.0721, # set as best learning rate from above
                                                                depth = 10, # tree depth, can tune
                                                                objective = "reg:squarederror"), # binary class problem 
                                                  data = x, 
                                                  nrounds = y,
                                                  silent = TRUE)),
    xg_train_mse = map2_dbl(xg_model, train_mat, xg_error, metric = "mse"), # train MSE
    xg_test_mse = map2_dbl(xg_model, test_mat, xg_error, metric = "mse") # test MSE
  )

# Calculating mean MSE for different nrounds
red_wine_xg_reg_nrounds_meanmse <- red_wine_xg_reg_nrounds %>%
  group_by(nrounds) %>%
  summarise(mean_train_mse = mean(xg_train_mse),
            mean_test_mse = mean(xg_test_mse))

ggplot(red_wine_xg_reg_nrounds_meanmse) + 
  geom_line(aes(nrounds, mean_train_mse, color = "Training Error")) +
  geom_line(aes(nrounds, mean_test_mse, color = "Test Error")) +
  scale_color_manual("", values = c("blue", "red")) + 
  labs(x = "Number of Trees", y = "MSE")

#Save the 3 best nrounds
(candidate_nrounds <- red_wine_xg_reg_nrounds_meanmse %>%
    arrange((mean_test_mse)) %>%
    head(3) %>%
    select(nrounds))

#300 trees

# candidate models --------------------------------------------------------

# Applying the candidate models
set.seed(4)
(red_wine_xg_reg <- tibble(
  train = train %>% list(),
  test = test %>% list() # importing test data
) %>%
    crossing(eta = candidate_etas$eta) %>% # crossing by 3 etas
    crossing(nrounds = candidate_nrounds$nrounds) %>% # crossing by 3 nrounds
    mutate(
      train_mat = map(train, xgb_matrix, outcome = "quality", exclude_vars = "id"), # transforming to xgb matrix
      test_mat = map(test, xgb_matrix, outcome = "quality", exclude_vars = "id"), # transforming to xgb matrix
      xg_model = pmap(.l = list(x = train_mat, y = eta, z = nrounds), # using pmap function
                      .f = function(x, y, z) xgb.train(params = list(eta = y, # set learning rate
                                                                     depth = 10, # tree depth, can tune
                                                                     objective = "reg:squarederror"), # binary class problem 
                                                       data = x, 
                                                       nrounds = z,
                                                       silent = TRUE)),
      xg_train_mse = map2_dbl(xg_model, train_mat, xg_error, metric = "mse"), # train MSE
      xg_test_mse = map2_dbl(xg_model, test_mat, xg_error, metric = "mse") # test MSE
    ) %>% 
    arrange(xg_test_mse) %>% # arranging by test MSE
    select(eta, nrounds, xg_train_mse, xg_test_mse)) # selecting relevant columns

# Storing the best model
red_wine_reg_models <- red_wine_xg_reg %>%
  select(eta, nrounds, xg_test_mse) %>%
  head(1) %>%
  mutate(model_type = "XGBoost", tuning = "eta = 0.0217, nrounds = 220") %>%
  rename(testMSE = xg_test_mse) %>%
  select(model_type, tuning, testMSE)

red_wine_reg_models %>%
  kable()

#From the results, it seems that eta is the more important determining tuning parameter than nrounds. 
#eta = 0.0217 seems to be the best one, and we can really probably any nround around the 160~220 range.
#Test MSE 0.381
#We will just pick the best one from this result and store it to compare with models from Random Forest.

# Random Forrest ----------------------------------------------------------
#We will fit the model and calculate the MSE using mtry from 1 to 15, again using 10-fold CV of the model building wildfires_train dataset. 
#In addition to the test and train MSE, we will also calculate OOB error rate. 
#When constructing tree from bootstrapped samples, a given observation gets left out from about 1/3 of the trees. 
#These observations are known as “out of bag,” since they were not used in those specific 1/3 of the trees, they are run through those trees, and the prediction error for those observations are calculated as well. 
#It’s important data to compare, but test_mse still holds more weight in terms of determining accuracy of models.
# Helper function
mse_ranger <- function(model, test, outcome){
  # Check if test is a tibble
  if(!is_tibble(test)){
    test <- test %>% as_tibble()
  }
  # Make predictions
  preds <- predict(model, test)$predictions
  # Compute MSE
  mse <- mean((test[[outcome]] - preds)^2)
  return(mse)
}

#Tuning mtry values and oob
set.seed(4)
red_wine_rf_mtry <- train %>% 
  crossv_kfold(10, id = "folds") %>%
  crossing(mtry = 1:(ncol(train) - 2)) %>% 
  mutate(
    train = map(train, as_tibble), 
    test = map(test, as_tibble), 
    #each value of mtry
    model = map2(.x = train, .y = mtry, 
                 .f = function(x, y) ranger(quality ~ . - id, 
                                            mtry = y, 
                                            data = x, 
                                            splitrule = "variance", 
                                            importance = "impurity")), 
    train_err = map2_dbl(model, train, mse_ranger, outcome = "quality"),
    test_err = map2_dbl(model, test, mse_ranger, outcome = "quality"), 
    oob_err = map_dbl(.x = model, 
                      .f = function(x) x[["prediction.error"]]) #this is the out of bag error
  )

(red_wine_rf_mtry_meanerror <- red_wine_rf_mtry %>%
    group_by(mtry) %>%
    summarise(mean_train_err = mean(train_err),
              mean_test_err = mean(test_err),
              mean_oob_err = mean(oob_err)) %>%
    arrange(mean_test_err))

ggplot(red_wine_rf_mtry_meanerror) + 
  geom_line(aes(mtry, mean_oob_err, color = "OOB Error")) +
  geom_line(aes(mtry, mean_train_err, color = "Training Error")) +
  geom_line(aes(mtry, mean_test_err, color = "Test Error")) + 
  labs(x = "mtry", y = "mean MSE") + 
  scale_color_manual("", values = c("purple", "red", "blue")) + 
  theme_bw()
#The OOB and Test Error are pretty similar, while the training error is low as expected. 
#The mean MSE seems tkeeps declining from mtry = 1, hit the lowest between 4.5-6, and increase again very slowly.
#Save best mtry
(candidate_mtry <- red_wine_rf_mtry_meanerror %>%
    arrange(mean_test_err) %>%
    head(3) %>%
    select(mtry))

### Applying the candidate models to test
(red_wine_rf_reg <- tibble(
  train = train %>% list(), 
  test = test %>% list() 
) %>%
    crossing(mtry = candidate_mtry$mtry) %>% 
    mutate(
      model = map2(.x = train, .y = mtry, 
                   .f = function(x, y) ranger(quality ~ . - id, 
                                              mtry = y, 
                                              data = x, 
                                              splitrule = "variance", 
                                              importance = "impurity")), # fitting models
      train_err = map2_dbl(model, train, mse_ranger, outcome = "quality"), 
      test_err = map2_dbl(model, test, mse_ranger, outcome = "quality"), 
      oob_err = map_dbl(.x = model, 
                        .f = function(x) x[["prediction.error"]])
    ) %>%
    arrange(test_err) %>%
    select(mtry, train_err, test_err, oob_err)) 

red_wine_reg_models <- red_wine_rf_reg %>%
  select(test_err) %>%
  head(1) %>%
  mutate(model_type = "Random Forest", tuning = "mtry = 5") %>%
  rename(testMSE = test_err) %>%
  select(model_type, tuning, testMSE) %>%
  rbind(red_wine_reg_models)

red_wine_reg_models %>%
  kable()


#MTRY 5 Test Error 0.341

#I decided not to do Bagging


# mtry analysis -----------------------------------------------------------
#QUE BUEN FUCKING GRAPH
red_wine_rf_reg_mtry5 <- ranger(quality ~ . - id, 
                                 data = test, 
                                 mtry = 5,
                                 importance = "impurity", 
                                 splitrule = "variance")
mtry_graph <- vip(red_wine_rf_reg_mtry5)
ggsave("mtry_graph.png")

#We will see how ALCOHOL affects QUALITY.
partial(red_wine_rf_reg_mtry5, 
        pred.var = "alcohol",
        pred.data = test,
        plot = TRUE,
        rug = TRUE,
        plot.engine = "ggplot2") + 
  labs(y = "Quality", x = "Alcohol")


#The upward trend is quite clear. 
#The sharpest decline happens between 10-12. 

# Linear Regression -------------------------------------------------------

model_lm_def <- tibble(modelNo = c("mod01", "mod02", "mod03", "mod04", "mod05",
                                   "mod06", "mod07", "mod08"), 
                       fmla = c("quality ~ . - id",
                                "quality ~ alcohol + sulphates + volatile_acidity + chlorides + p_h",
                                "quality ~ alcohol + sulphates + volatile_acidity + chlorides",
                                "quality ~ alcohol + sulphates + volatile_acidity",
                                "quality ~ alcohol + sulphates",
                                "quality ~ alcohol",
                                "quality ~ sulphates + volatile_acidity + chlorides",
                                "quality ~ alcohol + sulphates + chlorides"))

red_wine_lm_models <- train %>% 
  crossv_kfold(10, id = "folds") %>%
  crossing(model_lm_def) %>% # trying 10 different models
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble),
    # Fit models for each formula
    model_fit = map2(fmla, train, lm), 
    train_mse = map2_dbl(model_fit, train, mse), 
    test_mse = map2_dbl(model_fit, test, mse)
  )

(red_wine_lm_models_meanmse <- red_wine_lm_models %>%
    group_by(modelNo, fmla) %>%
    summarise(mean_train_mse = mean(train_mse),
              mean_test_mse = mean(test_mse)) %>%
    arrange(mean_test_mse))

# Storing Best 3 Models
(candidate_lm <- red_wine_lm_models_meanmse %>%
    arrange(mean_test_mse) %>%
    head(3) %>%
    select(modelNo, fmla))

#Mod 1 Test MSE:  0.436
# Fit Candidate Models
(red_wine_lm_reg <- tibble(
  train = train %>% list(),
  test = test %>% list()
) %>%
    crossing(candidate_lm) %>%
    mutate(
      # Fit models for each formula
      model_fit = map2(fmla, train, lm), 
      train_mse = map2_dbl(model_fit, train, mse), # train MSE
      test_mse = map2_dbl(model_fit, test, mse) # test MSE
    ) %>%
    arrange(test_mse) %>% # order from lowest to highest test MSE
    select(modelNo, fmla, train_mse, test_mse)) # select relevant columns

red_wine_reg_models <- red_wine_lm_reg %>%
  select(test_mse) %>%
  head(1) %>%
  mutate(model_type = "Linear", tuning = "fmla = quality ~ . - id") %>%
  rename(testMSE = test_mse) %>%
  select(model_type, tuning, testMSE) %>%
  rbind(red_wine_reg_models)

red_wine_reg_models %>%
  kable()

# Ridge -------------------------------------------------------------------

#200 possible lambda values 
lambda_grid <- 10^seq(-2, 10, length = 200)

#10 fold cross
red_wine_ridge_cv <- train %>% 
  cv.glmnet(
    formula = quality ~ . - id, 
    data = ., 
    alpha = 0, 
    nfolds = 10,
    lambda = lambda_grid
  )

# ridge's best lambdas
lambda_min <- red_wine_ridge_cv$lambda.min 
lambda_1se <- red_wine_ridge_cv$lambda.1se

#Sotring values
(lambdas <- tibble(lambda = c(lambda_min, lambda_1se)))

#Applying the candidate models to test
(red_wine_ridge_reg <- tibble(
  train = train %>% list(),
  test  = test %>% list()
) %>%
    mutate(
      ridge_min = map(train, ~ glmnet(quality ~ . - id, data = .x,
                                      alpha = 0, lambda = lambda_min)),
      ridge_1se = map(train, ~ glmnet(quality ~ . - id, data = .x,
                                      alpha = 0, lambda = lambda_1se)) 
    ) %>% 
    pivot_longer(cols = c(-test, -train), names_to = "method", values_to = "fit") %>% 
    mutate(pred = map2(fit, test, predict),
           test_mse = map2_dbl(test, pred, ~ mean((.x$quality - .y)^2))) %>% 
    arrange(test_mse) %>% 
    select(method, test_mse) %>% 
    cbind(lambdas) %>% 
    select(method, lambda, test_mse)) 

# Storing for Comparison
red_wine_reg_models <- red_wine_ridge_reg %>%
  select(test_mse) %>%
  head(1) %>%
  mutate(model_type = "Ridge", tuning = "lambda = 0.026") %>%
  rename(testMSE = test_mse) %>%
  select(model_type, tuning, testMSE) 

red_wine_reg_models %>%
  rbind()

#Ridge Min Test MSE: 0.4039920
#To my surprise, Linear and Ridge regression models seem to do a better job than the tree-based methods for predicting burned. Linear was actually slightly better than rigde, which I really did not expect.

# Summary of All Models
red_wine_reg_models %>%
  arrange(testMSE) %>%
  kable()
