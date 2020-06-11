
# white wine --------------------------------------------------------------

# Data wrangling 
library(tidyverse)
library(janitor)
library(modelr)
library(rsample)
library(knitr)
library(tidyselect)

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
white_wine <- read.csv("winequality-white.csv", 
                     sep=";", header=TRUE) %>%
  as_tibble() %>%
  clean_names()

## 75% of the sample size
## set the seed to make your partition reproducible
set.seed(4)
white_wine$id <- 1:nrow(white_wine)
train <- white_wine %>% dplyr::sample_frac(.75)
test  <- dplyr::anti_join(white_wine, train, by = 'id')

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
white_wine_rf_mtry <- train %>% 
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

(white_wine_rf_mtry_meanerror <- white_wine_rf_mtry %>%
    group_by(mtry) %>%
    summarise(mean_train_err = mean(train_err),
              mean_test_err = mean(test_err),
              mean_oob_err = mean(oob_err)) %>%
    arrange(mean_test_err))

ggplot(white_wine_rf_mtry_meanerror) + 
  geom_line(aes(mtry, mean_oob_err, color = "OOB Error")) +
  geom_line(aes(mtry, mean_train_err, color = "Training Error")) +
  geom_line(aes(mtry, mean_test_err, color = "Test Error")) + 
  labs(x = "mtry", y = "mean MSE") + 
  scale_color_manual("", values = c("purple", "red", "blue")) + 
  theme_bw()
#The OOB and Test Error are pretty similar, while the training error is low as expected. 
#The mean MSE seems tkeeps declining from mtry = 1, hit the lowest between 4.5-6, and increase again very slowly.
#Save best mtry
(candidate_mtry <- white_wine_rf_mtry_meanerror %>%
    arrange(mean_test_err) %>%
    head(3) %>%
    select(mtry))

### Applying the candidate models to test
(white_wine_rf_reg <- tibble(
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

white_wine_reg_models <- white_wine_rf_reg %>%
  select(test_err) %>%
  head(1) %>%
  mutate(model_type = "Random Forest", tuning = "mtry = 5") %>%
  rename(testMSE = test_err) %>%
  select(model_type, tuning, testMSE) %>%
  rbind(white_wine_reg_models)


# graph -------------------------------------------------------------------

white_wine_rf_reg_mtry4 <- ranger(quality ~ . - id, 
                                data = test, 
                                mtry = 4,
                                importance = "impurity", 
                                splitrule = "variance")
vip(white_wine_rf_reg_mtry4)

#We will see how ALCOHOL affects QUALITY.
partial(white_wine_rf_reg_mtry4, 
        pred.var = "alcohol",
        pred.data = test,
        plot = TRUE,
        rug = TRUE,
        plot.engine = "ggplot2") + 
  labs(y = "Quality", x = "Alcohol")


