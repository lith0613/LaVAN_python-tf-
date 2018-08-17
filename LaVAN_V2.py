import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

K.set_session(sess)

from keras          import metrics

from keras.applications.vgg16    import decode_predictions
from keras.utils.np_utils        import to_categorical
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image

import keras.backend     as K
import numpy             as np
from numpy.linalg import norm

import time


def predict_array(model, array, source_idx, target_idx):
    prediction = model.predict(array)
    _, category, proba = decode_predictions(prediction)[0][0]

    return  np.argmax(prediction), category, proba, prediction[0][source_idx], prediction[0][target_idx]


def generate_adversarial_examples(model, img_path, epsilon=5, source_idx=None, target_idx=None, size=299):

    K.set_learning_phase(0)

    x = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    preprocessed_array = preprocess_input(x)
    preprocessed_array[0, 210:260, 210:260, :] = preprocess_input(np.random.uniform(0, 255, (1, 50, 50, 3)))
    current_perturbation = np.zeros((1, 50, 50, 3))

    target = to_categorical(target_idx, 1000)
    target_variable = K.variable(target, dtype=tf.float32)
    source = to_categorical(source_idx, 1000)
    source_variable = tf.Variable(source, dtype=tf.float32)

    init_new_vars_op = tf.variables_initializer([target_variable, source_variable])
    sess.run(init_new_vars_op)

    class_variable_t = target_variable
    loss_func_t = metrics.categorical_crossentropy(model.output.op.inputs[0], class_variable_t)
    get_grad_values_t = K.function([model.input], K.gradients(loss_func_t, model.input))

    class_variable_s = source_variable
    loss_func_s = metrics.categorical_crossentropy(model.output.op.inputs[0], class_variable_s)
    get_grad_values_s = K.function([model.input], K.gradients(loss_func_s, model.input))

    cnt = 0

    # 300 --> the max number of iterations, change it if necessary

    while(cnt < 300):

        start = time.time()

        # Prints relevant info every iteration
        print(preprocessed_array[0, 2, 2, 0], predict_array(model, preprocessed_array, source_idx, target_idx), cnt)

        grad_values_t, grad_values_s = get_grad_values_t([preprocessed_array]), get_grad_values_s([preprocessed_array])

        diff = grad_values_t[0] - grad_values_s[0]

        perturbation_update =  - diff * epsilon

        current_perturbation += perturbation_update[:, 210:260, 210:260, :]

        preprocessed_array[0, 210:260, 210:260, :] = current_perturbation

        end = time.time()
        # Prints time it costs each iteration
        print(end - start)

        cnt += 1


    return preprocessed_array


def test_model():
    model = InceptionV3()

    for each in os.listdir("../Playground/Dataset/Sea_Lion/"):
        if(each.endswith('jpg')):
            size = 299
            img_path = "../Playground/Dataset/Sea_Lion/" + each
            x = image.load_img(img_path, target_size=(size, size))
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            preprocessed_array = preprocess_input(x)
            print(predict_array(model, preprocessed_array, 0, 0))

# Basically
def perturb_all(dir_path):
    model = InceptionV3()
    total, success = 0, 0
    for each in os.listdir(dir_path):
        if(each.endswith('jpg')):
            img_path = dir_path + each
            x = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            preprocessed_array = preprocess_input(x)

            tar_idx = 429
            src_idx, _, _, _, _ = predict_array(model, preprocessed_array, 0, 0)
            preprocessed_array = generate_adversarial_examples(model, img_path, 5, src_idx, tar_idx, 299)
            src_idx_2, _, _, _, _ = predict_array(model, preprocessed_array, 0, 0)
            if(src_idx_2 == tar_idx):
                success += 1
            total += 1

    print(success, total, round(success/total, 3))


# Loads Inception model by default
def perturb_one(img_path):
    model = InceptionV3()
    x = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    preprocessed_array = preprocess_input(x)

    tar_idx = random.randint(0, 999)
    src_idx, _, _, _, _ = predict_array(model, preprocessed_array, 0, 0)
    preprocessed_array = generate_adversarial_examples(model, img_path, 0.01, src_idx, tar_idx, 299)

    # Save the perturbation region if you like (by uncomment the line below)
    # image.array_to_img(preprocessed_array[0, 210:260, 210:260, :]).save(img_path[:-4] +"_what.jpg")


perturb_one("Target.jpg")
