


#Q1
rm(list=ls())
library(keras)
library(tensorflow)
#Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()
#load tensorflow backend to objectt K
K <- backend()
# Data Preparation -----------------------------------------------------
batch_size <- 128
num_classes <- 10
epochs <- 12
# Input image dimensions
img_rows <- 28
img_cols <- 28
# The data, shuffled and split between train and test sets
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
x_train <- train_images
x_test <- test_images
# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)
# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255
# Convert class vectors to binary class matrices
y_train <-  train_labels
y_test <- test_labels
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', padding = "same",
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu' , padding = "same") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu', padding = "same" , name = "Conv_last")%>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = num_classes, activation = 'softmax')
summary(model)

#specify optimizer, loss function, metrics

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 3,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)
#running optimization
history <- model %>% fit(
  x_train, y_train, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2,
  callbacks = callback_specs
)

#load the saved best model
model_best = load_model_hdf5('best_model.hdf5',compile=FALSE)
#compute the predicted values
p_hat_test = model_best %>% predict(x_test)
y_hat_test = apply(p_hat_test,1,which.max)
#evaluate the model performance
model %>% evaluate(x_test, y_test)


last_conv_layer <- model_best %>% get_layer("Conv_last") 
test_case_to_look=45 #Look at Cases #1, #45, #1001 , #1005 and #5666 in test data
target_output <- model_best$output[, which.max(y_test[test_case_to_look,])] 

grads <- K$gradients(target_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(1L, 2L))
compute_them <- K$`function`(list(model_best$input), 
                             list(pooled_grads, last_conv_layer$output[1,,,])) 

which.max(y_test[test_case_to_look,])

which.max(p_hat_test[test_case_to_look,])


#The input image has to be a 4D array
x_test_example <- x_test[test_case_to_look,,,]
dim(x_test_example) <- c(1,dim(x_test_example),1)

###Computing the importance and gradient map for each filter
c(pooled_grads_value, conv_layer_output_value) %<-% compute_them(list(x_test_example))
###Computing the Acivation Map
for (i in 1:128) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

#Normalizing the acivation map
heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)




y_true = apply(y_test,1,which.max)
sum(y_hat_test==y_true)/length(y_true) #do we get the same result?

### Demonstrating the result: Draw the Activation Map over the Original Image
add.alpha <- function(COLORS, ALPHA){
  if(missing(ALPHA)) stop("provide a value for alpha between 0 and 1")
  RGB <- col2rgb(COLORS, alpha=TRUE)
  RGB[4,] <- round(RGB[4,]*ALPHA)
  NEW.COLORS <- rgb(RGB[1,], RGB[2,], RGB[3,], RGB[4,], maxColorValue = 255)
  return(NEW.COLORS)
}


image(x_test[test_case_to_look,,,],col=grey.colors(100))
pal <- colorRampPalette(c(rgb(1,1,1), rgb(0,0,1)))
COLORS <- add.alpha(pal(20), 0.4)
image(heatmap,add=TRUE,col=COLORS)











#Q2 VNN

rm(list=ls())
data = load("~/DATA HW 2/diabetes_data.Rdata")

keras::backend()$clear_session()
# Normalize training data
train_df <- scale(train_data) 
# Normalize test data
col_means_train <- attr(train_df, "scaled:center") 
col_stddevs_train <- attr(train_df, "scaled:scale")
test_df <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

library(keras)
#Adapted from https://keras.rstudio.com/articles/examples/variational_autoencoder.html
#Modified for Minist data ... 
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()
library(tensorflow)
# Parameters --------------------------------------------------------------
batch_size <- 100L
original_dim <- 10L
latent_dim <- 2L
intermediate_dim <- 64L
epochs <- 200L
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
decoder_p <- layer_dense(units = 1, activation = "linear")
h_decoded <- decoder_h(z)
y_decoded_p <- decoder_p(h_decoded)
# we instantiate these layers separately so as to reuse them later
# end-to-end variational model
vnn <- keras_model(x, y_decoded_p)
# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

vnn_loss <- function(x, x_decoded_mean){
  cat_loss <- loss_mean_squared_error(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  cat_loss + kl_loss
}

vnn %>% compile(optimizer = "rmsprop", loss = vnn_loss , metrics = c("mse"))

#set up early stopping
#callback_specs=list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 10,
#                                            verbose = 0, mode = "auto"),
#                   callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
#)




# Model training ----------------------------------------------------------
#Model training ----------------------------------------------------------
  vnn %>% fit(
    train_df, train_y, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size, 
   
    validation_data = list(test_df, test_y))
#  callbacks = callback_specs)


#x_test_decoded <- array(NA,dim=c(221,1000))

#for(i in 1:1000){
 # x_test_decoded[,i] <- predict(vnn, test_df)}

x_train_decoded <- array(NA,dim=c(221,1000))

for(i in 1:1000){
  x_train_decoded[,i] <- predict(vnn, train_df)}

se <- sqrt(mean((x_train_decoded-train_y)^2))


#find 95% prediction interval
mean_pred <- apply(x_train_decoded,1,mean)

#PI <- rbind(mean_pred - 1.96*sd_pred,mean_pred + 1.96*sd_pred)
PI <- rbind(mean_pred - 1.96*se,mean_pred + 1.96*se)


library(ggplot2)
ggplot() +
  geom_point(mapping=aes(x=train_y,y=mean_pred)) +
  geom_segment(aes(x = train_y, y = PI[1,], xend = train_y, yend = PI[2,])) + 
  geom_segment(aes(x = train_y+1, y = PI[1,], xend = train_y+1, yend = PI[1,])) + 
  geom_segment(aes(x = train_y+1, y = PI[2,], xend = train_y+1, yend = PI[2,])) + 
  geom_abline()

##compute the empirical coverage
sum(PI[1,]<train_y & PI[2,]>train_y) / nrow(train_data)
#mean PI length
mean(PI[2,]- PI[1,])







#MC DROPOUT
# Setting up tuning parameters 
rm(list=ls())
data = load("~/DATA HW 2/diabetes_data.Rdata")
keras::backend()$clear_session()
# Normalize training data
train_df <- scale(train_data) 
# Normalize test data
col_means_train <- attr(train_df, "scaled:center") 
col_stddevs_train <- attr(train_df, "scaled:scale")
test_df <- scale(test_data, center = col_means_train, scale = col_stddevs_train)
DropoutRate <- 0.85
tau <- 0.1
keep_prob <- 1-DropoutRate
n_train <- nrow(train_df)
penalty_weight <- keep_prob/(tau* n_train) 
penalty_intercept <- 1/(tau* n_train)

#Setting up drouput from the beginning
dropout_1 <- layer_dropout(rate = DropoutRate)  
dropout_2 <- layer_dropout(rate = DropoutRate) 
dropout_3 <- layer_dropout(rate = DropoutRate) 



inputs = layer_input(shape = list(10))
output <- inputs %>%
  layer_dense(units = 256, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>% 
  dropout_1(training=TRUE) %>%
  layer_dense(units = 128, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_2(training=TRUE) %>%
  layer_dense(units = 64, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_3(training=TRUE) %>%
  layer_dense(units = 1, activation = 'linear',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept))
model <- keras_model(inputs, output)
summary(model)

#specify optimizer, loss function, metrics
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mse')
)
#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)
#running optimization
history <- model %>% fit(
  train_df, train_y, 
  epochs = 100, batch_size = 200, 
  validation_split = 0.2,
  callbacks = callback_specs
)

#load the saved best model
model_best = load_model_hdf5('best_model.hdf5',compile=FALSE)
#prediction via mcdropout sampling
mc.sample=1000
testPredict=array(NA,dim=c(nrow(test_data),mc.sample))
for(i in 1:mc.sample)
{
  testPredict[,i]=model_best %>% predict(test_df)
}

#find 95% prediction interval
mean_pred <- apply(testPredict,1,mean)
sd_pred <- sqrt(apply(testPredict,1,var)+1/tau)
PI <- rbind(mean_pred - 1.96*sd_pred,mean_pred + 1.96*sd_pred)
library(ggplot2)
ggplot() +
  geom_point(mapping=aes(x=test_y,y=mean_pred)) +
  geom_segment(aes(x = test_y, y = PI[1,], xend = test_y, yend = PI[2,])) + 
  geom_segment(aes(x = test_y-1, y = PI[1,], xend = test_y+1, yend = PI[1,])) + 
  geom_segment(aes(x = test_y-1, y = PI[2,], xend = test_y+1, yend = PI[2,])) + 
  geom_abline()
##compute the empirical coverage
sum(PI[1,]<test_y & PI[2,]>test_y) / nrow(test_data)
#mean PI length
mean(PI[2,]- PI[1,])




#QUANTILE REGRESSION
rm(list=ls())
data = load("~/DATA HW 2/diabetes_data.Rdata")
keras::backend()$clear_session()
# Normalize training data
train_df <- scale(train_data) 
# Normalize test data
col_means_train <- attr(train_df, "scaled:center") 
col_stddevs_train <- attr(train_df, "scaled:scale")
test_df <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

test_df <- array(test_df,dim=dim(test_df))
train_df <- array(train_df,dim=dim(train_df))
# Setting up tuning parameters 
DropoutRate <- 0.5
tau <- 0.1
#Define the model
keep_prob <- 1-DropoutRate
n_train <- nrow(train_df)
penalty_weight <- keep_prob/(tau* n_train) 
penalty_intercept <- 1/(tau* n_train) 
inputs = layer_input(shape = list(10))
output <- inputs %>%
  layer_dense(units = 256, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), 
              bias_regularizer=regularizer_l2(penalty_intercept)) %>% 
  layer_dropout(rate = DropoutRate) %>% 
  layer_dense(units = 128, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), 
              bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  layer_dropout(rate = DropoutRate) %>% 
  layer_dense(units = 64, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), 
              bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  layer_dropout(rate = DropoutRate) %>% 
  layer_dense(units = 1, activation = 'linear')


#Computing 0.50th quantile
quantile <- 0.5
tilted_loss <- function(q, y, f) {
  e <- y - f
  k_mean(k_maximum(q * e, (q - 1) * e) , axis = -1)
}
model <- keras_model(inputs, output)
model %>% compile(
  optimizer = 'adam',
  loss = function(y_true, y_pred)
    tilted_loss(quantile, y_true, y_pred),
  metrics = "mae"
)
summary(model)
#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", 
                                            min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' 
                                              ,save_best_only = TRUE)
)
#running optimization
history <- model %>% fit(
  train_df, train_y, 
  epochs = 300, 
  validation_split = 0.2,
  callbacks = callback_specs
)
#load the saved best model
model_best = load_model_hdf5('best_model.hdf5',compile=FALSE)
pred_median = model_best %>% predict(test_df)

#Computing 0.975th quantile
quantile <- 0.975
model975 <- keras_model(inputs, output)
model975 %>% compile(
  optimizer = 'adam',
  loss = function(y_true, y_pred)
    tilted_loss(quantile, y_true, y_pred),
  metrics = "mae"
)

#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, 
                                            patience = 10,
                                            verbose = 0, mode = "auto"),
                    
                    callback_model_checkpoint(filepath='model975.hdf5',save_freq='epoch' 
                                              ,save_best_only = TRUE)
)
#running optimization
history <- model975 %>% fit(
  train_df, train_y, 
  epochs = 300, 
  validation_split = 0.2,
  callbacks = callback_specs
)
#load the saved best model
model975 = load_model_hdf5('model975.hdf5',compile=FALSE)
pred975 = model975 %>% predict(test_df)

#Computing 0.025th quantile
quantile <- 0.025
model025 <- keras_model(inputs, output)
model025 %>% compile(
  optimizer = 'adam',
  loss = function(y_true, y_pred)
    tilted_loss(quantile, y_true, y_pred),
  metrics = "mae"
)

##set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, 
                                            patience = 10,
                                            verbose = 0, mode = "auto"),
                    
                    callback_model_checkpoint(filepath='model025.hdf5',save_freq='epoch' 
                                              ,save_best_only = TRUE)
)
#running optimization
history <- model025 %>% fit(
  train_df, train_y, 
  epochs = 300, 
  validation_split = 0.2,
  callbacks = callback_specs
)
#load the saved best model
model025 = load_model_hdf5('model025.hdf5',compile=FALSE)
pred025 = model025 %>% predict(test_df)

#performance evaluation
library(ggplot2)
ggplot() +
  geom_point(mapping=aes(x=test_y,y=pred_median)) +
  geom_segment(aes(x = test_y, y =pred025, xend = test_y, yend = pred975)) + 
  geom_segment(aes(x = test_y-1, y = pred025, xend = test_y+1, yend = pred025)) + 
  geom_segment(aes(x = test_y-1, y = pred975, xend = test_y+1, yend = pred975)) + 
  geom_abline()
##compute the empirical coverage
sum(c(pred025)<test_y & c(pred975)>test_y) / nrow(test_data)

#mean PI length
mean(pred975- pred025)

