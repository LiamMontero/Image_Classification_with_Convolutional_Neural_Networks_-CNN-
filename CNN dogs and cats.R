#las siguientes lineas de codigo se ejecutan en BASH
#curl -L -o ~/Downloads/microsoft-catsvsdogs-dataset.zip\
#https://www.kaggle.com/api/v1/datasets/download/shaunthesheep/microsoft-catsvsdogs-dataset

#descomprimeindo el archivo
#unzip ~/Downloads/microsoft-catsvsdogs-dataset.zip -d ~/Downloads/microsoft-catsvsdogs-dataset

original_dataset_dir_cat <- "~/Downloads/microsoft-catsvsdogs-dataset/PetImages/Cat"
original_dataset_dir_dog <- "~/Downloads/microsoft-catsvsdogs-dataset/PetImages/Dog"
base_dir <- "~/Downloads/cats_and_dogs_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0(1:1000, ".jpg")
file.copy(file.path(original_dataset_dir_cat, fnames),
          file.path(train_cats_dir))
fnames <- paste0(1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir_cat, fnames),
          file.path(validation_cats_dir))
fnames <- paste0(1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir_cat, fnames),
          file.path(test_cats_dir))
fnames <- paste0(1:1000, ".jpg")
file.copy(file.path(original_dataset_dir_dog, fnames),
          file.path(train_dogs_dir))
fnames <- paste0(1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir_dog, fnames),
          file.path(validation_dogs_dir))
fnames <- paste0(1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir_dog, fnames),
          file.path(test_dogs_dir))


#costruyendo el modelo
library(keras3)
model_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)


# convirtiendo imagenes a tensores
train_dataset <- image_dataset_from_directory(
  directory = normalizePath(train_dir),
  image_size = c(150, 150),
  batch_size = 20,
  label_mode = "binary",
  labels = "inferred"
)

validation_dataset <- image_dataset_from_directory(
  directory = normalizePath(validation_dir),
  image_size = c(150, 150),
  batch_size = 20,
  label_mode = "binary",
  labels = "inferred"
)

# Normalización (rescale)
normalize <- layer_rescaling(scale = 1/255)
train_dataset <- train_dataset$map(function(x, y) list(normalize(x), y))
validation_dataset <- validation_dataset$map(function(x, y) list(normalize(x), y))


history <- model_1 %>% fit(
  train_dataset,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_dataset,
  validation_steps = 50
)

model_1 %>% save_model("cats_and_dogs_small_1.h5")


#generando mas imagenes para mitigar el sobre ajuste(aumentando los datos)
# Secuencia de capas de aumento
data_augmentation_pipeline <- keras_model_sequential(
  name = "data_augmentation",
  layers = list(
    # 1. Reescalado (equivalente a rescale = 1/255)
    #layer_rescaling(scale = 1/255, offset = 0.0, input_shape = c(150, 150, 3L)), # no hago el rescalado porque ya lo hice anteriormente en los datos
    # 2. Volteo Horizontal Aleatorio (equivalente a horizontal_flip = TRUE)
    layer_random_flip(mode = "horizontal", input_shape = c(150, 150, 3L)),
    # 3. Rotación Aleatoria (equivalente a rotation_range = 40)
    layer_random_rotation(
      factor = 40/360, # Rango de [-40, +40] grados. factor es fracción de 2*pi
      fill_mode = "nearest",
      interpolation = "bilinear"
    ),
    # 4. Traslación Aleatoria (equivalente a width_shift_range = 0.2, height_shift_range = 0.2)
    layer_random_translation(
      height_factor = 0.2, # Fracción de la altura total para el desplazamiento
      width_factor = 0.2,  # Fracción del ancho total para el desplazamiento
      fill_mode = "nearest",
      interpolation = "bilinear"
    ),
    # 5. Cizallamiento Aleatorio (equivalente a shear_range = 0.2)
    # (Asumiendo que layer_random_shear existe y funciona como se espera)
    layer_random_shear(
      x_factor = 0.2, # Intensidad del cizallamiento (a menudo en radianes)
      fill_mode = "nearest",
      interpolation = "bilinear"
    ),
    # 6. Zoom Aleatorio (equivalente a zoom_range = 0.2)
    layer_random_zoom(
      height_factor = 0.2, # Rango de zoom [1-0.2, 1+0.2] -> [0.8, 1.2]
      width_factor = 0.2,  # Puede ser el mismo o diferente para ancho
      fill_mode = "nearest",
      interpolation = "bilinear"
    )
  )
)


model_2 <- keras_model_sequential() %>%
  data_augmentation_pipeline %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)

history_2 <- model_2 %>% fit(
  train_dataset,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_dataset,
  validation_steps = 50
)

model_2 %>% save_model("cats_and_dogs_small_2.h5")