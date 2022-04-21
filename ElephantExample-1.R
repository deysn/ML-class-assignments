library(tensorflow)
library(keras)
K <- backend()

tf$compat$v1$disable_eager_execution()

##Load Vgg16, a pre-constructed convolutional layers
model <- application_vgg16(weights = "imagenet") 

#Download a image of elephants from the web 
image <- get_file("elephant.jpg", "https://goo.gl/zCTWXW") %>% 
  image_load(target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(image)
imagenet_decode_predictions(preds, top = 3)[[1]]

# class_name class_description      score
# 1  n02504458  African_elephant 0.78988522
# 2  n01871265            tusker 0.19872670
# 3  n02504013   Indian_elephant 0.01114247

which.max(preds[1,])

# [1] 387

african_elephant_output <- model$output[, 387]
last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- K$gradients(african_elephant_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(1L, 2L))
iterate <- K$`function`(list(model$input), 
                        list(pooled_grads, last_conv_layer$output[1,,,])) 

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(image))

for (i in 1:512) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)


###Creating the Plot
##Warning: No need to understand each line!

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)


write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap(heatmap, "elephant_heatmap.png") 

library(magick) 
library(viridis) 

img_path="https://raw.githubusercontent.com/rstudio/keras/master/vignettes/examples/creative_commons_elephant.jpg"

image <- image_read(img_path) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "elephant_overlay.png", 
              width = 14, height = 14, bg = NA, col = pal_col) 


###########################
## Creating the Fian Image
###########################
image_read("elephant_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot() 