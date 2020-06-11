
# white clusters ----------------------------------------------------------

library(ISLR)
library(kernlab)
library(gridExtra)
library(ggdendro)
library(magrittr)
library(janitor)
library(skimr)
library(tidyverse)
library(parameters)
library(conflicted)
library(cluster)
library(splus2R)
library(fpc)

# managing conflicts 
conflict_prefer("filter", "dplyr")
conflict_prefer("alpha", "kernlab")
conflict_prefer("combine", "gridExtra")
conflict_prefer("extract", "magrittr")
conflict_prefer("map", "purrr")

#load data
white_wine <- read.csv("winequality-white.csv", 
                     sep=";", header=TRUE) %>%
  clean_names()

set.seed(100) # setting seed
white_wine$id <- 1:nrow(white_wine)
train <- white_wine %>% dplyr::sample_frac(.75)
test  <- dplyr::anti_join(white_wine, train, by = 'id')


#default dist is Euclidian
help_hclust <- function(data, meth){
  data <- data %>% select(-id)
  return(hclust(dist(data), method = meth))
}

#hf to cut the dendogram
cut_hclust <- function(hclust_obj, ncuts){
  return(cutree(hclust_obj, ncuts))
}

#scaling the data- this is gonna help for part c
scale_data <- function(df) {
  id <- df %>% select(id)
  df_scaled <- df %>% select(-id) %>% scale() %>% as_tibble()
  return(bind_cols(df_scaled, id))
}

white_wine_hclust <- tibble(data = list(train,
                                      train %>% scale_data())) %>% 
  mutate(hclust = map(data, help_hclust, meth = 'complete'),
         graph = map(hclust, ggdendrogram),
         clusters = map(hclust, cut_hclust, ncuts = 3)) 


#Cut the dendrogram at a height that results in three distinct clusters. 
#Which states belong to which clusters?
white_wine_hclust %>% 
  pluck('graph', 1)

unscaled_clusters <- white_wine_hclust %>% pluck('data', 1) %>% 
  bind_cols(cluster = white_wine_hclust %>% pluck('clusters', 1))

# K-Means Clustering with 5 clusters
fit <- kmeans(unscaled_clusters, 5)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(unscaled_clusters, fit$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(unscaled_clusters, fit$cluster)

# library(mclust)
# fit <- Mclust(unscaled_clusters)
# plot(fit) # plot results 
# summary(fit) # display the best model

#Hierarchically cluster the states using complete linkage and Euclidean distance,
#after scaling the variables to have standard deviation one.

scaled_clusters <- white_wine_hclust %>% pluck('data', 2) %>% 
  bind_cols(cluster = white_wine_hclust %>% pluck('clusters', 2))

#plot us map

#### Exercise 2 - KMEANS

#cluster SS
get_within_ss <- function(kmean_obj){
  return(kmean_obj$tot.withinss)
}

#cluster labels for the data
lab_cluster <- function(x, clust_obj){
  
  if(class(clust_obj) == "kmeans"){
    clust = clust_obj$cluster
  } else {
    clust = clust_obj
  }
  
  out = x %>% 
    mutate(cluster = clust)
  
  return(out)
}

white_wine_categorical <- white_wine %>% 
  select(-quality) %>% 
  onehot::onehot() %>% 
  predict(white_wine) %>% as_tibble()

white_wine_scaled <- white_wine %>% 
  select_if(is.numeric) %>% 
  scale() %>% as_tibble()

white_wine_one_hot <- white_wine_scaled %>% 
  bind_cols(white_wine_categorical)

add_id <- function(df){
  id <- white_wine %>% select(id)
  cluster_with_id <- df %>% bind_cols(id)
  return(cluster_with_id)
}

add_id <- function(df){
  id <- white_wine %>% select(id)
  cluster_with_id <- df %>% bind_cols(id)
  return(cluster_with_id)
}

white_wine_clustering <- tibble(data = list(white_wine_one_hot)) %>% 
  crossing(k = seq(2, 6, 1)) %>% 
  mutate(k_mean = map2(data, k, kmeans, nstart = 20), # to reintialize the rand start 20 times
         within_clust_ss = map_dbl(k_mean, get_within_ss), # extract the ss within each cluster
         cluster = map2(data, k_mean, lab_cluster), # to get the label for the cluster
         cluster = map(.x = cluster,
                       .f = function(x) {
                         cluster_fac <- x %>% mutate(cluster = as.factor(cluster))
                         return(cluster_fac)
                       }),
         cluster_with_id = map(cluster, add_id))

#PCA on original data to plot 2D tensor
pca_out <- white_wine %>% 
  select_if(is.numeric) %>% 
  prcomp(scale = TRUE)


two_principle_components <- pca_out$x[,1] %>% enframe(name = NULL) %>% 
  rename(PC1 = value) %>% 
  bind_cols(pca_out$x[,2] %>% enframe(name = NULL) %>% 
              rename(PC2 = value))

#adding pc to df
add_pc <- function(df) {
  df_components <- df %>% bind_cols(two_principle_components)
  return(df_components)
}

#plotting principle components and clusters
plot_cluster <- function(df) {
  plot <- df %>% 
    ggplot(aes(x = PC1, y = PC2, color = cluster)) + 
    geom_point(alpha = 0.4)
  return(plot)
}

white_wine_clustering <- white_wine_clustering %>% 
  mutate(white_wine_pc = map(cluster_with_id, add_pc),
         cluster_plots = map(white_wine_pc, plot_cluster)) 

white_wine_clustering %>% 
  pluck('cluster_plots') %>% 
  grid.arrange(grobs = .)

#if k = 4, there's 4 seperable groups

white_wine_clustering %>% 
  ggplot(aes(x = k, y = within_clust_ss)) + 
  geom_line()


plot_kmeans <- white_wine_clustering %>% 
  filter(k == 4) %>% 
  select(cluster_plots) %>% 
  rename(plots = cluster_plots)



run_hclust_white_wine <- function(x, meth){
  return(hclust(dist(x), method = meth))
}

#hierarchical clustering
hc_data <- tibble(data = list(white_wine_one_hot)) %>% 
  crossing(k = seq(2, 6, 1)) %>% 
  mutate(hclust = map(data, run_hclust_white_wine, meth = 'complete'),
         cluster = map2(hclust, k, cut_hclust),
         cluster = map(cluster, as.factor),
         cluster_data = map2(data, cluster, lab_cluster),
         cluster_addid = map(cluster_data, add_id),
         cluster_pc = map(cluster_addid, add_pc),
         plots = map(cluster_pc, plot_cluster)) # plots of the clusters

hc_data %>% 
  pluck('plots') %>% 
  grid.arrange(grobs = .)


plot_hierarchical <- hc_data %>% 
  filter(k == 4) %>% 
  select(plots) 

# spec_clust <- tibble(data = list(white_wine_one_hot)) %>% 
#   mutate(spec = map(.x = data, 
#                     .f = function(x) specc(as.matrix(x), centers = 4)))
#   
# 
# spec_clust <- spec_clust %>% 
#   mutate(cluster_data = map2(data, spec, lab_cluster),
#          cluster = map(.x = cluster_data,
#                        .f = function(x) {
#                          cluster_fac <- x %>% mutate(cluster = as.factor(cluster))
#                          return(cluster_fac)
#                        }),
#          cluster_addid = map(cluster, add_id),
#          cluster_pc = map(cluster_addid, add_pc),
#          plots = map(cluster_pc, plot_cluster))
# 
# spec_clust %>% 
#   pluck('plots')

# plot_spectral <- spec_clust %>% 
#   select(plots) 


#dont use the one-hot encoded data
mixed_clusters <- tibble(data = list(white_wine)) %>% 
  mutate(dissimilarity = map(data, daisy), # daisy returns a matrix of dissim of the data points
         cluster = map(dissimilarity, pam, k = 3))


cluster_data <- mixed_clusters %>% pluck('cluster')

clusters_data <- white_wine %>% bind_cols(cluster_data[[1]]$clustering %>% enframe(name = NULL) %>% 
                                          rename(cluster = value)) %>% 
  mutate(cluster = as_factor(cluster))

mixed_clusters <- tibble(data = list(clusters_data)) %>% 
  mutate(cluster_pc = map(data, add_pc),
         plots = map(cluster_pc, plot_cluster))

mixed_clusters %>% 
  pluck('plots')
