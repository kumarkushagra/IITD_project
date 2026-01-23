# import tensorflow as tf
# from tensorflow.keras import layers
# import os

# IMG_SIZE = (224, 224)

# data_augmentation = tf.keras.Sequential([
#     layers.RandomRotation(0.083),      # ~15°
#     layers.RandomTranslation(0.1, 0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomFlip("horizontal"),
# ])



# def load_image(path):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_image(img, channels=3, expand_animations=False)
#     img = tf.image.resize(img, IMG_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0
#     return img

# def load_dataset(dataset_dir="/home/vector/master_dataset",batch_size=32,val_split=0.2,seed=42):
#     dataset_dir = os.path.abspath(dataset_dir)
#     class_names = sorted(os.listdir(dataset_dir))
#     class_indices = {name: i for i, name in enumerate(class_names)}

#     files = []
#     labels = []

#     for cls in class_names:
#         cls_dir = os.path.join(dataset_dir, cls)
#         for f in os.listdir(cls_dir):
#             if f.lower().endswith((".jpg", ".png", ".jpeg")):
#                 files.append(os.path.join(cls_dir, f))
#                 labels.append(class_indices[cls])

#     files = tf.constant(files)
#     labels = tf.one_hot(labels, depth=len(class_names))

#     # shuffle once
#     idx = tf.random.shuffle(tf.range(len(files)), seed=seed)
#     files = tf.gather(files, idx)
#     labels = tf.gather(labels, idx)

#     split = int(len(files) * (1 - val_split))

#     train_files, val_files = files[:split], files[split:]
#     train_labels, val_labels = labels[:split], labels[split:]

#     def make_ds(f, y, training):
#         ds = tf.data.Dataset.from_tensor_slices((f, y))
#         # IMP PART 
#         ds = ds.map(
#             lambda p, l: (load_image(p), l),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#         if training:
#             ds = ds.map(
#                 lambda x, y: (data_augmentation(x, training=True), y),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#             ds = ds.shuffle(1024)
#         ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return ds

#     train_ds = make_ds(train_files, train_labels, training=True)
#     val_ds   = make_ds(val_files, val_labels, training=False)

#     return train_ds, val_ds, class_indices

# ---------------- OLD CODE USING ImageDataGenerator (DEPRECATED) ----------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(dataset_dir="/home/vector/master_dataset",img_size=(224, 224),batch_size=32,val_split=0.2,seed=42):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen, train_gen.class_indices
