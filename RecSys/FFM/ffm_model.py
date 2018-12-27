import numpy as np
import pandas as pd
import tensorflow as tf
import os 

#先确定各参数量的值
#数据量大小，field, features, 

all_data_size = 1000
field_nums = 2
input_features = 20
vec_size = 3
lr = 0.01
batch_size = 1#使用sgd

def gen_data():
    labels = [-1, 1]
    y = [np.random.choice(labels,1)[0] for _ in range(all_data_size)]
    x_field = [i//10 for i in range(input_features)]
    x = np.random.randint(0,2, size=(all_data_size, input_features))
    return x, y, x_field

def createLinearWeight(input_features):
    weights = tf.truncated_normal([input_features])
    tf_weights = tf.Variable(weights)
    return tf_weights

def createFFMWeight(input_features, field_nums, vec_size):
    weights = tf.truncated_normal([input_features, field_nums, vec_size])
    tf_weights =  tf.Variable(weights)
    return tf_weights

def createBias(input_features):
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)
    return tf_weights

def inference(input_x, input_fields, bias, linearWeight, ffmWeight):
    #线性
    linear_part = tf.add(tf.reduce_sum(tf.multiply(linearWeight, input_x)), bias)

    ffm_part = tf.Variable(0.0, dtype=tf.float32)

    for i in range(input_features):
        featureIndex1 = i
        fieldIndex1 = int(input_fields[i])
        for j in range(i+1, input_features):
            featureIndex2 = j
            fieldIndex2 = int(input_fields[j])
            #ffm part
            '''
            latent1 = tf.gather(ffmWeight, [featureIndex1, fieldIndex2], axis=0)
            #print(latent1)
            latent1 = tf.squeeze(latent1)
            
            latent2 = tf.gather(ffmWeight, [featureIndex2, fieldIndex1], axis=0)
            latent2 = tf.squeeze(latent2)
            latent_mul = tf.reduce_sum(tf.multiply(latent1, latent2))
            '''
            vectorLeft = tf.convert_to_tensor([[featureIndex1,fieldIndex2,i] for i in range(vec_size)])
            weightLeft = tf.gather_nd(ffm_weights,vectorLeft)
            weightLeftAfterCut = tf.squeeze(weightLeft)

            vectorRight = tf.convert_to_tensor([[featureIndex2,fieldIndex1,i] for i in range(vec_size)])
            weightRight = tf.gather_nd(ffm_weights,vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut,weightRightAfterCut))

            xi = tf.squeeze(tf.gather(input_x, [i]))
            xj = tf.squeeze(tf.gather(input_x, [j]))
            #print(xi)
            product = tf.reduce_sum(tf.multiply(xi, xj))
            ffm_part += tf.multiply(tempValue, product)

    combine_part = tf.add(linear_part, ffm_part)
    return combine_part

if __name__ == '__main__':
    trainx, trainy, input_fields = gen_data()

    input_x = tf.placeholder(tf.float32, [input_features])
    input_y = tf.placeholder(tf.float32)

    lambda_w = tf.constant(0.001, name="lambda_w")
    lambda_v = tf.constant(0.001, name="lambda_v")


    bias = createBias(input_features)
    linear_weights = createLinearWeight(input_features)
    ffm_weights = createFFMWeight(input_features, field_nums, vec_size)

    y_ = inference(input_x, input_fields, bias, linear_weights, ffm_weights)

    l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(linear_weights, 2)), 
        tf.reduce_sum(tf.multiply(lambda_v, tf.pow(ffm_weights, 2)),axis=[1,2])))

    loss = tf.log(1+tf.exp(input_y*y_)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(all_data_size):
            for t in range(all_data_size):
                input_x_batch = trainx[t]
                input_y_batch = trainy[t]
                predict_loss,_ = sess.run([loss,train_step],
                                               feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=i, predict_loss=predict_loss))

                #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                #writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                #writer.close()
























