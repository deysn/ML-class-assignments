


#CNN
rm(list=ls())
library(keras)
library(tensorflow)
data = load("C:/Users/chatt/OneDrive - University of Cincinnati/Desktop/Machine Leanring/Project/deers_frogs_trucks.Rdata")

#Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()
#load tensorflow backend to object K
K <- backend()
batch_size <- 256
num_classes <- 3
epochs <- 100

# Input image dimensions
img_rows <- 32
img_cols <- 32

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 3))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 3))
y_train = array_reshape(y_train, c(nrow(y_train), 3))
input_shape <- c(img_rows, img_cols, 3)
inputs = layer_input(shape = input_shape)


# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices
#y_train <- to_categorical(y_train, num_classes)
#y_test <- to_categorical(y_test, num_classes)


cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Define model


model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',padding = "same",input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu',padding = "same") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu',padding = "same", name = "Conv_last") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    
  layer_flatten() %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3)%>%
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3)%>%
  layer_dense(units = num_classes, activation = 'softmax')

summary(model)

#specify optimizer, loss function, metrics
model%>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='C:/Users/chatt/OneDrive - University of Cincinnati/Desktop/Machine Leanring/Project/best_model1.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)

#running optimization
history <- model %>% fit(
  x_train, y_train, 
  epochs = 60, batch_size = batch_size, 
  validation_split = 0.2,
  callbacks = callback_specs
)


#load the saved best model
model_best = load_model_hdf5('C:/Users/chatt/OneDrive - University of Cincinnati/Desktop/Machine Leanring/Project/best_model1.hdf5',compile=T)


#evaluate the model performance
model_best%>% evaluate(x_test, y_test)



test_case_to_look <- 1000 #Look at Case 33, 52, and 248 in test data
number_of_filters <- 64 #number of filers for hte last layer
K <- backend()
#Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()
###Define the functions
last_conv_layer <- model_best %>% get_layer("Conv_last") 
#Note: "conv2d_1" part need to be changed if the name of the layer is changed!
#Keras changes the names of layers every time the model is defined. compute the predicted values
p_hat_test = model_best%>% predict(x_test)
y_hat_test = apply(p_hat_test,1,which.max)

target_output <- model_best$output[, which.max(y_test[test_case_to_look,])] 
grads <- K$gradients(target_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(1L, 2L))
compute_them <- K$`function`(list(model_best$input), 
                             list(pooled_grads, last_conv_layer$output[1,,,])) 
#The input image has to be a 4D array
x_test_example <- x_test[test_case_to_look,,,]
dim(x_test_example) <- c(1,dim(x_test_example))
#True Label
which.max(y_test[test_case_to_look,])
#Original Label
which.max(model_best %>% predict(x_test_example))
###Computing the importance and gradient map for each filter
c(pooled_grads_value, conv_layer_output_value) %<-% compute_them(list(x_test_example))
###Computing the Acivation Map
for (i in 1:number_of_filters) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
#Normalizing the acivation map
heatmap <- pmax(heatmap, 0) 
heatmap <- (heatmap - min(heatmap))/ (max(heatmap)-min(heatmap)) 
library(magick)
library(viridis) 
write_heatmap <- function(heatmap, filename, width = 32, 
                          height = 32,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

image <- image_read(x_test[test_case_to_look,,,]) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue =255)
write_heatmap(heatmap, "overlay.png", 
              width = 32, height = 32, bg = NA, col = pal_col) 


image %>% plot()
image_read("overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "35") %>%
  plot()






#VNN
rm(list=ls())
library(keras)
library(tensorflow)
data = load("C:/Users/chatt/OneDrive - University of Cincinnati/Desktop/Machine Leanring/Project/deers_frogs_trucks.Rdata")

#Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()
#load tensorflow backend to objectt K
K <- backend()
batch_size <- 128
num_classes <- 3
epochs <- 100

# Input image dimensions
img_rows <- 32
img_cols <- 32

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 3))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 3))
y_train = array_reshape(y_train, c(nrow(x_train), 3))
input_shape <- c(img_rows, img_cols, 3)


# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices
#y_train <- to_categorical(y_train, num_classes)
#y_test <- to_categorical(y_test, num_classes)


cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')


latent_dim <- 8L

epsilon_std <- 1.0



# Define model
inputs <-  layer_input(shape =input_shape)

output = inputs%>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu',padding = "same") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) 



z_mean <- layer_dense(output, latent_dim)
z_log_var <- layer_dense(output, latent_dim)
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
z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)


#decoder from Z to Y
decoder_h <- layer_dense(units = 128, activation = "relu")
decoder_p <- layer_dense(units = 3, activation = "softmax")
h_decoded <- decoder_h(z)
y_decoded_p <- decoder_p(h_decoded)
# we instantiate these layers separately so as to reuse them later
# end-to-end variational model
vnn <- keras_model(inputs, y_decoded_p)
# encoder, from inputs to latent space
encoder <- keras_model(inputs, z_mean)

vnn_loss <- function(inputs, y_decoded_p){
  cat_loss <- loss_categorical_crossentropy(inputs, y_decoded_p)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  cat_loss + kl_loss
}
vnn %>% compile(optimizer = "adam", loss = vnn_loss)


summary(vnn)
# Model training ----------------------------------------------------------
vnn %>% fit(
  x_train, y_train, 
  shuffle = TRUE, 
  epochs = 20, 
  batch_size = 128, 
  validation_data = list(x_test, y_test)
)

P = vnn%>% predict(x_test)

h1 = hist(apply(P,1,which.max))

h2 = hist(apply(y_test,1,which.max))

plot( h1, col=rgb(0,0,1,1/4))  # first histogram
plot( h2, col=rgb(1,0,0,1/4),add=T)  # second



# Visualizations ----------------------------------------------------------
library(ggplot2)
library(dplyr)
x_test_encoded <- predict(vnn, x_test, batch_size = 
                            batch_size)
x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(y_test)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()



#MCDROPOUT
rm(list=ls())
library(keras)
library(tensorflow)
data = load("C:/Users/chatt/OneDrive - University of Cincinnati/Desktop/Machine Leanring/Project/deers_frogs_trucks.Rdata")

#Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()
#load tensorflow backend to objectt K
K <- backend()
batch_size <- 128
num_classes <- 3
epochs <- 100

# Input image dimensions
img_rows <- 32
img_cols <- 32

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 3))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 3))
y_train = array_reshape(y_train, c(nrow(x_train), 3))
input_shape <- c(img_rows, img_cols, 3)


# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices
#y_train <- to_categorical(y_train, num_classes)
#y_test <- to_categorical(y_test, num_classes)


cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')


DropoutRate <- 0.5
tau <- 0.5
keep_prob <- 1-DropoutRate
n_train <- nrow(x_train)
penalty_weight <- keep_prob/(tau* n_train) 
penalty_intercept <- 1/(tau* n_train)

#Setting up drouput from the beginning
dropout_1 <- layer_dropout(rate = DropoutRate)  
dropout_2 <- layer_dropout(rate = DropoutRate) 
dropout_3 <- layer_dropout(rate = DropoutRate) 

# Define model
inputs <-  layer_input(shape =input_shape)

output = inputs%>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu',padding = "same",
                kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu',padding = "same",
                kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_2(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu',kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_3(training=TRUE) %>%
  layer_dense(units = 3, activation = 'softmax',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept))
model <- keras_model(inputs, output)
summary(model)


#specify optimizer, loss function, metrics
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)
#set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)
#running optimization
history <- model %>% fit(
  x_train, y_train, 
  epochs = 100, batch_size = 200, 
  validation_split = 0.2,
  callbacks = callback_specs
)

#load the saved best model
model_best = load_model_hdf5('best_model.hdf5',compile=FALSE)

#evaluate the model performance
model %>% evaluate(x_test, y_test)


P = model_best %>% predict(x_test)

h1 = hist(apply(P,1,which.max))

h2 = hist(apply(y_test,1,which.max))

plot( h1, col=rgb(0,0,1,1/4))  # first histogram
plot( h2, col=rgb(1,0,0,1/4), add=T)  # second
