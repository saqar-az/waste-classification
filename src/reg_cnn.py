import os
import re
import numpy as np
import ast
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import non_max_suppression
from sklearn.model_selection import train_test_split
import shutil
import random
import json  

original_ds = 'original_dataset'
split_ds = 'split_dataset'
img_h, img_w = 224, 224
train_sec = 0.7
val_sec = 0.15
test_sec = 0.15
rndm = 42
ckpt_dir = 'checkpoints'   
best_model_path = 'best_model.h5'  
best_hp_file = 'best_hyperparams.txt'
aug_train = True   
aug_num = 5  # 5 aug per img
aug_dir = 'train_augmented' 
search_progress_file = 'search_progress.json'

class_name = ['battery', 'biological', 'cardboard', 'clothes', 'glass','metal', 'paper', 'plastic', 'shoes', 'trash']
usable = {'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes'}
not_usable = {'battery', 'biological', 'trash'}

def split_dataset():
    if os.path.exists(os.path.join(split_ds, 'train')) and os.listdir(os.path.join(split_ds, 'train')) != []:
        print("dataset split already exists")
        return
    
    class_dirs = []
    for name in os.listdir(original_ds):
        path = os.path.join(original_ds, name)
        if os.path.isdir(path):      
            class_dirs.append(name)      
    class_dirs.sort()    
    print(f"found classes: {class_dirs}")    

    for split in ['train','val','test']:
        for cls in class_dirs:
            os.makedirs(os.path.join(split_ds, split, cls), exist_ok=True)

    random.seed(rndm)

    for cls in class_dirs:
        src = os.path.join(original_ds, cls)
        imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg'))]
        train_val, test = train_test_split(imgs, test_size=test_sec, random_state=rndm)
        val_sec_adj = val_sec / (train_sec + val_sec)
        train, val = train_test_split(train_val, test_size=val_sec_adj, random_state=rndm)
        # copy from org to split
        for f in train:
            shutil.copy2(os.path.join(src, f), os.path.join(split_ds,'train',cls,f))
        for f in val:
            shutil.copy2(os.path.join(src, f), os.path.join(split_ds,'val',cls,f))
        for f in test:
            shutil.copy2(os.path.join(src, f), os.path.join(split_ds,'test',cls,f))
        print(f"{cls}: train={len(train)}, val={len(val)}, test={len(test)}")
    print("dataset split done\n")

def train_aug():
    src_train_dir = os.path.join(split_ds, 'train')
    aug_train_dir = os.path.join(split_ds, aug_dir)

    if os.path.exists(aug_train_dir) and os.listdir(aug_train_dir) != []:
        print(f"augmented training folder already exists")
        return aug_train_dir

    print("\ntrain data augmentation")
    print(f"creating {aug_num} new images per original in {aug_train_dir}")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for cls in class_name:
        cls_src = os.path.join(src_train_dir, cls)
        cls_dst = os.path.join(aug_train_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)

        img_files = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg'))] # org imgs
        print(f"class '{cls}': {len(img_files)} originals: {len(img_files) * aug_num} augmented + {len(img_files)} originals")

        for fname in img_files:
            img_path = os.path.join(cls_src, fname)
            img = cv2.imread(img_path) # BGR 
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB for tf
            img_rgb = img_rgb.astype('float32')  # float for tf
            img_batch = np.expand_dims(img_rgb, axis=0) # reshape to (H,W,C,1) for generator
            aug_iter = datagen.flow(img_batch, batch_size=1, # generate augmented images
                                    save_to_dir=cls_dst,
                                    save_prefix=f'aug_{os.path.splitext(fname)[0]}',
                                    save_format='jpg')

            for _ in range(aug_num):
                next(aug_iter)
            shutil.copy2(img_path, os.path.join(cls_dst, fname)) # copy org imgs to aug 

    print(f"augmentation complete\n")
    return aug_train_dir

# def cnn(input_shape=(224, 224, 3), num_classes=10):
#     model = models.Sequential([
#         layers.Conv2D(32,3, activation='relu', input_shape=input_shape),
#         layers.Conv2D(32,3, activation='relu'),
#         layers.MoaxPoling2D(2),
#         layers.Dropout(0.25),

#         layers.Conv2D(64,3, activation='relu'),
#         layers.Conv2D(64,3, activation='relu'),
#         layers.MaxPooling2D(2),
#         layers.Dropout(0.25),

#         layers.Conv2D(128,3, activation='relu'),
#         layers.MaxPooling2D(2),
#         layers.Dropout(0.25),

#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x) # to work with Relu
    x = layers.BatchNormalization()(x) 
    x = layers.Activation('relu')(x) # same as ResNet firt batch norm then act
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        # makeing them same sizes
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def cnn(input_shape=(224, 224, 3), num_classes=10, filters_base=32, dropout_rate=0.5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters_base, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs) # make it ready for relu
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, filters_base)
    x = residual_block(x, filters_base)
    x = residual_block(x, filters_base*2, stride=2)
    x = residual_block(x, filters_base*2)
    x = residual_block(x, filters_base*4, stride=2)
    x = residual_block(x, filters_base*4)

    x = layers.GlobalAveragePooling2D()(x) # same size as channel nums
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x) # avoid overfiltting

    model = models.Model(inputs, outputs)
    return model

def hyperparameter_search(train_dir):
    search_data = [
        {'lr': 0.0005, 'batch_size': 16, 'dropout': 0.3, 'filters_base': 32},
        {'lr': 0.001, 'batch_size': 32, 'dropout': 0.5, 'filters_base': 32},
        {'lr': 0.01, 'batch_size': 64, 'dropout': 0.4, 'filters_base': 48},
        {'lr': 0.001, 'batch_size': 16, 'dropout': 0.3, 'filters_base': 48},
        {'lr': 0.0005, 'batch_size': 64, 'dropout': 0.5, 'filters_base': 32},
    ]
    search_epoch   = 12
    search_p = 5
    print("\nhyperparameter search")

    completed = set()
    best_val_loss = float('inf')
    best_hparams = None
    results = []

    if os.path.exists(search_progress_file):
        with open(search_progress_file, 'r') as f:     
            progress = json.load(f)
    
        saved_data = progress.get('hparamss', [])
        if len(saved_data) == len(search_data) and saved_data == search_data:   
            completed = set(progress.get('completed', []))
            best_val_loss = progress.get('best_val_loss', float('inf'))
            best_hparams = progress.get('best_hparams', None)
            results = progress.get('results', []) # (conf, val loss)
            print(f"resuming from {len(completed)}.")
        else:
            print("start from scratch")

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    for i, hparams in enumerate(search_data):
        if i in completed:
            print(f"skipping hparams {i+1}/{len(search_data)}")
            continue
        
        lr = hparams['lr']
        bs = hparams['batch_size']
        drop = hparams['dropout']
        filters = hparams['filters_base']

        print(f"\nhparams {i+1}/{len(search_data)}: {hparams}")

        train_gen = train_datagen.flow_from_directory(train_dir,target_size=(img_h, img_w), batch_size=bs,class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(os.path.join(split_ds, 'val'), target_size=(img_h, img_w), batch_size=bs, class_mode='categorical', shuffle=False)

        model = cnn(
            input_shape=(img_h, img_w, 3),
            num_classes=len(class_name),
            filters_base=filters,
            dropout_rate=drop
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        temp_path = f'temp_search_{i}.h5'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=search_p, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(temp_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        hist = model.fit(train_gen, validation_data=val_gen, epochs=search_epoch, callbacks=callbacks, verbose=2)
        val_loss = min(hist.history['val_loss'])
        results.append((hparams, val_loss))
        print(f"best val_loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hparams = hparams
            model.save('best_search_model.h5')

        if os.path.exists(temp_path):
            os.remove(temp_path)

        completed.add(i)

        progress = {
            'completed': list(completed),
            'best_val_loss': best_val_loss,
            'best_hparams': best_hparams,
            'results': [(dict(c), v) for c, v in results],
            'hparamss': search_data   
        }

        # add to json
        with open(search_progress_file, 'w') as f:
            json.dump(progress, f)

    # if os.path.exists(search_progress_file):
    #     os.remove(search_progress_file)

    print("\nsearch results")
    for hparams, vloss in sorted(results, key=lambda x: x[1]): # based on val loss (hparam, val loss)
        marker = "best" if hparams == best_hparams else ""
        print(f"{hparams}  val_loss={vloss:.4f}{marker}")

    with open(best_hp_file, 'w') as f:
        f.write(f"best = {best_hparams}\n")
        f.write(f"val_loss = {best_val_loss:.4f}\n")
    print(f"\nbest hparam saved to {best_hp_file}")
    return best_hparams

def last_ckpt():
    if not os.path.isdir(ckpt_dir):
        return None, 0
    max_ep = 0
    latest = None
    for f in os.listdir(ckpt_dir):
        m = re.match(r'model_epoch_(\d+)\.h5', f)
        if m:
            ep = int(m.group(1))
            if ep > max_ep:
                max_ep = ep
                latest = f
    if latest:
        return os.path.join(ckpt_dir, latest), max_ep
    return None, 0

def train(hparams, train_dir):
    epoch = 250
    # early_stopping_p = 12
    lr_p = 3
    lr_value = 0.5
    min_lr = 1e-7
    lr = hparams['lr']
    bs = hparams['batch_size']
    drop = hparams['dropout']
    filters = hparams['filters_base']

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,target_size=(img_h, img_w), batch_size=bs,class_mode='categorical', shuffle=True)
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(split_ds, 'val'),target_size=(img_h, img_w), batch_size=bs,class_mode='categorical', shuffle=False)

    resume_path, initial_epoch = last_ckpt()

    if resume_path:
        print(f"resuming from epoch {initial_epoch} checkpoint: {resume_path}")
        model = tf.keras.models.load_model(resume_path, compile=False) # compile with weights

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("starting final training from scratch")
        model = cnn(
            input_shape=(img_h, img_w, 3),
            num_classes=len(class_name),
            filters_base=filters,
            dropout_rate=drop
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'model_epoch_{epoch:02d}.h5'), save_weights_only=False, save_freq='epoch', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_p, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_value, patience=lr_p,min_lr=min_lr, verbose=1)
    ]

    history = model.fit(train_gen, validation_data=val_gen,epochs=epoch, initial_epoch=initial_epoch,callbacks=callbacks, verbose=1)
    model.save(best_model_path)
    print(f"best model saved to {best_model_path}")
    return model, history

def evaluate_test(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(os.path.join(split_ds, 'test'),target_size=(img_h, img_w), batch_size=32,class_mode='categorical', shuffle=False)
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"test accuracy: {acc:.4f}, test loss: {loss:.4f}")
    return loss, acc

def classify_single_image(model, image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_w, img_h))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0) # bs, h, w, c
    preds = model.predict(img_batch, verbose=0)[0] # cause bs is 1
    cls_idx = np.argmax(preds)                       
    confidence = preds[cls_idx]                      

    name = class_name[cls_idx]
    usability = "usable" if name in usable else "not usable"
    print(f"image: {image_path}")
    print(f"prediction: {usability}, {name}, confidence: {confidence:.4f}")


if __name__ == '__main__':
    split_dataset()
    if aug_train:
        train_dir = train_aug()
    else:
        train_dir = os.path.join(split_ds, 'train')

    if os.path.exists(best_model_path):
        print(f"loading existing best model weights from {best_model_path}")
        model = cnn(input_shape=(img_h, img_w, 3), num_classes=len(class_name)) # reload the model with best weights
        model.load_weights(best_model_path)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    else:
        if os.path.exists(best_hp_file):
            hparams = {}
            with open(best_hp_file, 'r') as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith('best'):
                    dict_str = line.split('=', 1)[-1].strip() # take the last part (actual data)
                    try:
                        hparams = ast.literal_eval(dict_str) # make it python dict
                    except (ValueError, SyntaxError):
                        print("could'nt read hyperparameters")
                        hparams = {}
                    break
            if hparams:
                print(f"loaded saved hyperparameters: {hparams}")
            else:
                print("could'nt load hyperparameters")
                os.remove(best_hp_file)
                hparams = hyperparameter_search(train_dir)
        else:
            hparams = hyperparameter_search(train_dir)

        print("\nfinal training")
        model, _ = train(hparams, train_dir)

    print("\nevaluating on test set")
    evaluate_test(model)

    test_image = 'split_dataset/test/clothes/clothes_6.jpg'  
    if os.path.exists(test_image):
        classify_single_image(model, test_image)
