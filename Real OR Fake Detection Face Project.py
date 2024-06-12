import os 
import shutil 
import tensorflow as tf 
import random 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam 
import zipfile 

zip_fil = 'real-and-fake-face-detection.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall()

real_data = 'real_and_fake_face/training_real'
fake_data = 'real_and_fake_face/training_fake'

# Creating and Handling Data Directory 

com_dir = 'Target_DIR'
os.makedirs(com_dir, exist_ok =True)
os.makedirs(os.path.join(com_dir, 'train', 'real'), exist_ok = True)
os.makedirs(os.path.join(com_dir, 'val', 'real'), exist_ok = True)
os.makedirs(os.path.join(com_dir, 'train', 'fake'), exist_ok = True)
os.makedirs(os.path.join(com_dir, 'val', 'fake'), exist_ok = True)

# Function for Image Preprocessing 
def split_dir(source_dir, train_dir, val_dir, split_ratio= 0.8):
    files = os.listdir(source_dir)
    num_fil = len(files)
    train_num = int(num_fil * split_ratio)
    random.shuffle(files)
    for i, fil in enumerate(files):
        source_path = os.path.join(source_dir, fil)
        if i < train_num:
            destination_path = os.path.join(train_dir, fil)
        else:
            destination_path = os.path.join(val_dir, fil)
        shutil.copyfile(source_path, destination_path)


#Image Preprocessing for real data
split_dir(
    real_data,
    os.path.join(com_dir, 'train', 'real'),
    os.path.join(com_dir, 'val', 'real')
)

#Image Preprocessing for fake data 
split_dir(
    fake_data,
    os.path.join(com_dir, 'train', 'fake'),
    os.path.join(com_dir, 'val', 'fake')
)

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range= 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    zoom_range = 0.5,
    shear_range= 0.2,
    fill_mode = 'nearest',
    rotation_range = 15
)

#Train Data 
train_data = datagen.flow_from_directory(
    os.path.join(com_dir, 'train'),
    target_size = (228, 228),
    batch_size = 64,
    class_mode = 'binary'
)

#Test Data 
val_data = datagen.flow_from_directory(
    os.path.join(com_dir, 'val'),
    target_size = (228, 228),
    batch_size = 64,
    class_mode = 'binary'
)

#Building and Developing Model Architecture 

Model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (228, 228, 3)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation ='relu'),
    tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation =  'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(1, activation = 'softmax')
])

#Compiling the model
Model.compile(loss = 'binary_crossentropy',
             optimizer = Adam(learning_rate = 0.0001),
             metrics = ['accuracy'])

            
#Fit the model
Model.fit(train_data, epochs = 20, validation_data = val_data)