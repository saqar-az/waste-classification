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
from sklearn.metrics import recall_score
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
aug_num = 5 # 5 aug per img
aug_dir = 'train_augmented'
hazard_weight = 3
search_progress_file = 'search_progress.json'
confidence_thresh = 0.8 # more restrict
iou_thresh = 0.3 # intersection over union, less better
max_reg = 300 # less regions 
class_name = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
usable = {'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes'}
not_usable = {'battery', 'biological', 'trash'}
hazard_classes = [class_name.index(c) for c in ['battery', 'biological', 'trash']]

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
            img_batch = np.expand_dims(img_rgb, axis=0) # reshape to (1,H,W,C) for generator
            aug_iter = datagen.flow(img_batch, batch_size=1, # generate augmented images
                                    save_to_dir=cls_dst,
                                    save_prefix=f'aug_{os.path.splitext(fname)[0]}',
                                    save_format='jpg')

            for _ in range(aug_num):
                next(aug_iter)
            shutil.copy2(img_path, os.path.join(cls_dst, fname)) # copy org imgs to aug 

    print(f"augmentation complete\n")
    return aug_train_dir

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

def task_fom(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, labels=hazard_classes, average=None) # recall for hazard_classes only
    return np.mean(recalls)   # higher better

class CustomMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen): # based on val
        super().__init__()
        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs=None):
        self.val_gen.reset()
        y_true = self.val_gen.classes
        y_pred = np.argmax(self.model.predict(self.val_gen, verbose=0), axis=1)
        fom = task_fom(y_true, y_pred)
        logs['val_fom'] = fom
        print(f"val_fom: {fom:.4f}")

def hyperparameter_search(train_dir):
    search_data = [
        {'lr': 0.0005, 'batch_size': 16, 'dropout': 0.3, 'filters_base': 32},
        {'lr': 0.001,  'batch_size': 32, 'dropout': 0.5, 'filters_base': 32},
        {'lr': 0.01,   'batch_size': 64, 'dropout': 0.4, 'filters_base': 48},
        {'lr': 0.001,  'batch_size': 16, 'dropout': 0.3, 'filters_base': 48},
        {'lr': 0.0005, 'batch_size': 64, 'dropout': 0.5, 'filters_base': 32},
    ]
    search_epoch = 12
    search_p = 5

    print("\nhyperparameter search (using FoM)")

    completed = set()
    best_fom = 0.0 # based on fom       
    best_hparams = None
    results = []

    if os.path.exists(search_progress_file):
        with open(search_progress_file, 'r') as f:
            progress = json.load(f)
        saved_data = progress.get('hparamss', [])
        if len(saved_data) == len(search_data) and saved_data == search_data:
            completed = set(progress.get('completed', []))
            best_fom = progress.get('best_fom', 0.0)
            best_hparams = progress.get('best_hparams', None)
            results = progress.get('results', [])
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

        train_gen = train_datagen.flow_from_directory(train_dir, target_size=(img_h, img_w), batch_size=bs, class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(os.path.join(split_ds, 'val'), target_size=(img_h, img_w), batch_size=bs, class_mode='categorical', shuffle=False)
        model = cnn(input_shape=(img_h, img_w, 3), num_classes=len(class_name), filters_base=filters, dropout_rate=drop)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        temp_path = f'temp_search_{i}.h5'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=search_p,restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(temp_path, monitor='val_loss',save_best_only=True, verbose=0)
        ]

        hist = model.fit(train_gen, validation_data=val_gen,epochs=search_epoch, callbacks=callbacks, verbose=2)

        val_gen.reset()
        y_true = val_gen.classes
        y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
        fom = task_fom(y_true, y_pred)
        results.append((hparams, fom))
        print(f" , task FoM = {fom:.4f}")

        if fom > best_fom:                    
            best_fom = fom
            best_hparams = hparams
            model.save('best_search_model.h5')
        if os.path.exists(temp_path):
            os.remove(temp_path)

        completed.add(i)
        progress = {
            'completed': list(completed),
            'best_fom': best_fom,
            'best_hparams': best_hparams,
            'results': results,
            'hparamss': search_data
        }

        # add to json
        with open(search_progress_file, 'w') as f:
            json.dump(progress, f)

    print("\nsearch results by FoM")
    for hparams, fom in sorted(results, key=lambda x: x[1], reverse=True): # based on fom (hparam, fom)
        marker = "best" if hparams == best_hparams else ""
        print(f"{hparams}  FoM={fom:.4f}{marker}")

    with open(best_hp_file, 'w') as f:
        f.write(f"best = {best_hparams}\n")
        f.write(f"best_task_fom = {best_fom:.4f}\n")
    print(f"\nbest hyperparameters saved to {best_hp_file}")
    return best_hparams

def train(hparams, train_dir):
    epoch = 300
    early_stopping_p = 12
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

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(img_h, img_w), batch_size=bs, class_mode='categorical', shuffle=True)
    val_gen = val_datagen.flow_from_directory(os.path.join(split_ds, 'val'), target_size=(img_h, img_w), batch_size=bs, class_mode='categorical', shuffle=False)

    class_weights = None
    if hazard_classes:
        class_weights = {}
        for idx, name in enumerate(class_name):
            if name in not_usable:         
                class_weights[idx] = hazard_weight # if not usable weight is set to 3
            else:
                class_weights[idx] = 1.0 # if usable weight is set to 1
        print(f"using class weights: {class_weights}")

    resume_path, initial_epoch = last_ckpt()
    if resume_path:
        print(f"resuming from epoch {initial_epoch}")
        model = tf.keras.models.load_model(resume_path, compile=False) # compile with weights
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("starting final training from scratch")
        model = cnn(input_shape=(img_h, img_w, 3), num_classes=len(class_name), filters_base=filters, dropout_rate=drop)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        CustomMetricCallback(val_gen),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'model_epoch_{epoch:02d}.h5'),save_weights_only=False, save_freq='epoch', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_fom', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_fom', patience=early_stopping_p, mode='max', restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_value, patience=lr_p, min_lr=min_lr, verbose=1)
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epoch, initial_epoch=initial_epoch, class_weight=class_weights,callbacks=callbacks, verbose=1)
    model.save(best_model_path)
    print(f"best model saved to {best_model_path}")
    return model, history

def evaluate_test(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(os.path.join(split_ds, 'test'), target_size=(img_h, img_w), batch_size=32, class_mode='categorical', shuffle=False)
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"test accuracy: {acc:.4f}, test loss: {loss:.4f}")

    test_gen.reset()  # fom
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    fom = task_fom(y_true, y_pred)
    print(f"test task FoM (hazard recall): {fom:.4f}")
    return loss, acc

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

def classify_single_image(model, image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_w, img_h))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0) # 1, h, w, filter
    preds = model.predict(img_batch, verbose=0)[0] 
    cls_idx = np.argmax(preds)                       
    confidence = preds[cls_idx]                      

    name = class_name[cls_idx]
    usability = "usable" if name in usable else "not usable"
    print(f"image: {image_path}")
    print(f"prediction: {usability}, {name}, confidence: {confidence:.4f}")

def generate_regions(img_bgr, max_proposals=max_reg):
    search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    search.setBaseImage(img_bgr) # set the base image as the input image
    search.switchToSelectiveSearchQuality()
    rects = search.process() # process the image
    return rects[:max_proposals]

def create_box(img_bgr, rects, img_w=224, img_h=224, min_ratio=0.1):
    rois = []
    boxes = []
    (H, W) = img_bgr.shape[:2]

    for (x, y, w, h) in rects:
        if w / float(W) < min_ratio or h / float(H) < min_ratio: # if roi smaller than 10% of img get rid of it
            continue

        roi = img_bgr[y:y + h, x:x + w] # extract the Roi from image 
        if roi.size == 0:
            continue
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # to rgb
        roi = cv2.resize(roi, (img_w, img_h))
        roi = roi.astype('float32') / 255.0

        rois.append(roi)
        x1, y1, x2, y2 = x, y, x + w, y + h # box cordinates
        boxes.append((x1, y1, x2, y2))

    return rois, boxes    

def predict_rois(model, rois):
    rois_array = np.array(rois) # rios to np array for model (w, h, c, number)
    preds = model.predict(rois_array, verbose=0) # predict all rios the same time 
    class_indices = np.argmax(preds, axis=1) # best indx for each rio
    confidences = np.max(preds, axis=1) # best value for each rio

    return class_indices, confidences

def build_objects(boxes, class_indices, confidences, confidence_thresh=confidence_thresh):
    objects = {}
    for box, cls_idx, conf in zip(boxes, class_indices, confidences):
        if conf < confidence_thresh:
            continue
        label = class_name[cls_idx]
        objects.setdefault(label, []).append((box, conf))
    return objects

def apply_nms(boxes, scores, score_thresh=0.0, nms_thresh=iou_thresh): # score_thresh=0.0 cause its done in build_objects
    cv2_boxes = []
    # nms needs x,y,w,h
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        cv2_boxes.append([int(x1), int(y1), int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(cv2_boxes, scores, score_thresh, nms_thresh) # output= box, score

    selected_indices = []
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple)) else i
            selected_indices.append(idx)

    return selected_indices


def apply_nms_per_class(objects, iou_thresh=iou_thresh): # based on class name
    final_detections = []

    for label, dets in objects.items():
        boxes = [d[0] for d in dets]  # list of (x1, y1, x2, y2)
        scores = [d[1] for d in dets]  # confidence scores

        selected = apply_nms(boxes, scores, score_thresh=0.0, nms_thresh=iou_thresh)

        for idx in selected:
            x1, y1, x2, y2 = boxes[idx]
            conf = scores[idx]
            final_detections.append((x1, y1, x2, y2, conf, label))

    return final_detections

# overapping diff classes
def apply_nms_global(detections, iou_thresh=iou_thresh):
    if not detections:
        return []
    boxes = [d[:4] for d in detections] # x1, y1, x2, y2
    scores = [d[4] for d in detections] # conf
    selected = apply_nms(boxes, scores, score_thresh=0.0, nms_thresh=iou_thresh)
    return [detections[i] for i in selected]


def multi_objects_detection(model, img_path, confidence_thresh=confidence_thresh, iou_thresh=iou_thresh, max_proposals=max_reg):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # rgb for model

    proposals = generate_regions(img_bgr, max_proposals)
    rois, boxes = create_box(img_bgr, proposals)

    if not rois:
        print("no valid regions found.")
        return img_rgb, []

    class_ids, confs = predict_rois(model, rois)
    objects = build_objects(boxes, class_ids, confs, confidence_thresh)
    final_detections = apply_nms_per_class(objects, iou_thresh)

    # global nms to remove overlapping boxes
    final_detections = apply_nms_global(final_detections, iou_thresh)

    # draw boxes 
    for (x1, y1, x2, y2, conf, label) in final_detections:
        color = (0, 255, 0) if label in usable else (255, 0, 0)
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2) # (x1,y1) top left ,(x2,y2) bottom right
        cv2.putText(img_rgb, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # on top of the box

    return img_rgb, final_detections    

if __name__ == '__main__':
    split_dataset()
    if aug_train:
        train_dir = train_aug()
    else:
        train_dir = os.path.join(split_ds, 'train')

    if os.path.exists(best_model_path):
        print(f"loading existing best model from {best_model_path}")
        try:
            model = tf.keras.models.load_model(best_model_path)
        except Exception:
            print("direct loading failed, rebuilding")
            model = cnn(input_shape=(img_h, img_w, 3), num_classes=len(class_name))
            model.load_weights(best_model_path)
    else:
        if os.path.exists(best_hp_file):
            with open(best_hp_file, 'r') as f:
                content = f.read()
            hparams = {}
            for line in content.splitlines():
                if line.startswith('best'):
                    dict_str = line.split('=', 1)[-1].strip() # take the last part
                    try:
                        hparams = ast.literal_eval(dict_str) # make it python dict
                    except (ValueError, SyntaxError):
                        print("couldn't read hyperparameters")
                        hparams = {}
                    break
            if hparams:
                print(f"loaded saved hyperparameters: {hparams}")
            else:
                print("couldn't load hyperparameters, running search")
                os.remove(best_hp_file)
                hparams = hyperparameter_search(train_dir)
        else:
            hparams = hyperparameter_search(train_dir)

        print("\nfinal training")
        model, _ = train(hparams, train_dir)

    print("\nevaluating on test set")
    evaluate_test(model)

    # test_image = 'split_dataset/test/clothes/clothes_6.jpg'  
    # if os.path.exists(test_image):
    #     classify_single_image(model, test_image)

    test_image_mo = 'mo.png' 
    result_img, detections = multi_objects_detection(model, test_image_mo)
    cv2.imshow('Detected Objects', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
