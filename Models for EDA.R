

```{r}
# Model in tibble
#gotta vectorize
model_def <- tibble(models = c("mod01", "mod02"), 
                    fmla = c("quality ~ . - quality_type", 
                             "quality ~ poly(alcohol, 3) + poly(volatile_acidity, 3) + poly(density, 3) + citric_acid"))
```

```{r 10-Fold-CV}
wine_10fold <- red_wine %>% 
  crossv_kfold(10, id = "fold") %>%
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

```{r Test-MSE}
(wine_10fold <- wine_10fold %>%
    crossing(model_def) %>% 
    mutate(
      model_fit = map2(fmla, train, lm),
      fold_mse = map2_dbl(model_fit, test, mse) # this calculates the test set's MSE
    ))
```

```{r average-per-model}
wine_10fold %>% 
  group_by(models) %>%
  summarise(avg_mse = mean(fold_mse)) %>%
  arrange(avg_mse) %>%
  kable 
```

Model 1's (sink model) Test MSE is much smaller than model 2's, so it should be the best option.

```{r helper-function}
#this function calculates error rate
error_rate_glm <- function(data, model){
  data %>% 
    mutate(pred_prob = predict(model, newdata = data, type = "response"),
           pred = if_else(pred_prob > 0.5, 1, 0),
           error = pred != wlf) %>% 
    pull(error) %>% 
    mean()
}
```

```{r model-in-tibble}
second_model_def <- tibble(models = c("mod01", "mod02"), 
                           fmla = c("quality_type ~ . - quality_type", 
                                    "quality_type ~ poly(alcohol, 2) + density + poly(volatile_acidity, 2) + poly(citric_acid, 3) + x*y"))
```

```{r ten-cross}
second_wine_10fold <- red_wine %>% 
  crossv_kfold(10, id = "fold") %>%
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

```{r}
(second_wine_10fold <- second_wine_10fold %>%
    crossing(second_model_def) %>% 
    mutate(
      model_fit = map2(fmla, train, family = binomial, glm), 
      misclass_rate_glm = map2_dbl(test, model_fit, error_rate_glm)
    ))
```

