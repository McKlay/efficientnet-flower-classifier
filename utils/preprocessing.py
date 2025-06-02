## Preprocessing inside app.py

'''
def predict_flower(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)  # raw [0, 255]
    img_array = preprocess_input(img_array)  # normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
'''

## make sure processing during inference (app,py) 
## is the same as preprocessing during training
## see code cell 3  in https://www.kaggle.com/code/claymarksarte/flower-recognition-fine-tuning

'''
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(image, [IMG_SIZE, IMG_SIZE, 3])

    image = tf.cast(image, tf.float32)  # keep as float32 but keep original [0,255] values
    image = preprocess_input(image)     # ✅ now safely normalize to [-1, 1]

    label = tf.cast(label, tf.int32)
    return image, label
'''