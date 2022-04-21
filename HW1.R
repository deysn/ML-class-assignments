library(keras)

boston_housing <- dataset_boston_housing()
c(train_x, train_y) %<-% boston_housing$train
c(test_x, test_y) %<-% boston_housing$test

library(tibble)
column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
train_df <- as_tibble(train_x)
colnames(train_df) <- column_names
# Normalize training data
train_df <- scale(train_df) 
# Normalize test data
col_means_train <- attr(train_df, "scaled:center") 
col_stddevs_train <- attr(train_df, "scaled:scale")
test_df <- scale(test_x, center = col_means_train, scale = col_stddevs_train)
model <- keras_model_sequential()
model %>%
  layer_dense(units =256, activation = 'relu', input_shape = c(13)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units =128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units =64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)
summary(model)

model %>% compile(
  loss = 'mse',
  optimizer = 'rmsprop',
  metrics = c('mae')
)

callback_specs=list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 50,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)

history <- model %>% fit(
  train_x, train_y, 
  epochs = 300, batch_size = 128, 
  validation_split = 0.2,
  callbacks = callback_specs
)
model_best = load_model_hdf5('best_model.hdf5',compile=FALSE)


#compute the predicted values
p_hat_test = model_best%>% predict(test_x)


#evaluate the model performance
model %>% evaluate(test_x, test_y)

plot(test_y,p_hat_test, xlab = "Oiginal data" , ylab = "Predicted values")

