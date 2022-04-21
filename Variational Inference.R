rm(list=ls())
keras::backend()$clear_session()

library(tensorflow)
library(keras)

#Adapted from https://keras.rstudio.com/articles/examples/variational_autoencoder.html
#Modified for Minist data ... 

if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(tensorflow)
# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 64L
epochs <- 20L
epsilon_std <- 1.0
# Model definition --------------------------------------------------------
x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>%   layer_lambda(sampling)
#decoder from Z to Y
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_p <- layer_dense(units = 10, activation = "softmax")
h_decoded <- decoder_h(z)
y_decoded_p <- decoder_p(h_decoded)



# we instantiate these layers separately so as to reuse them later
# end-to-end variational model
vnn <- keras_model(x, y_decoded_p)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)
vnn_loss <- function(x, x_decoded_mean){
  cat_loss <- loss_categorical_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  cat_loss + kl_loss
}

vnn %>% compile(optimizer = "adam", loss = vnn_loss)

# Data preparation --------------------------------------------------------
mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))


y_train <- to_categorical(mnist$train$y)
y_test <- to_categorical(mnist$test$y)

# Model training ----------------------------------------------------------

vnn %>% fit(
  x_train, y_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, y_test)
)
vnn %>% evaluate(x_test, y_test)
# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

library(GofCens)
gen_dat <- function(n, beta, sigma) {
  x <- rnorm(n)
  y <- 0 + beta*x + rnorm(n, 0, sigma)
  data.frame(x = x, y = y)
}

set.seed(1)
dat <- gen_dat(100, 0.30, 1)

mc <- lmcavi(dat$y, dat$x, tau2 = 0.50^2)
mc