
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()

#函数声明

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



#定义输入输出结构
xs=tf.placeholder(tf.float32,[None,28*28])

ys=tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32)
#x_image rehspae 成28*28*1的形状，因为是灰色图，前面-1那一位代表的是样本数量，-1指样本数量不确定
x_image=tf.reshape(xs,[-1,28,28,1])
#1，2两个参数是卷积核的尺寸大小，第三个参数是图像通道数，第四个参数是卷积核的数目
W_conv1=weight_variable([5,5,1,32])

b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#池化，卷积结果乘以池化结果
h_pool1=max_pool_2x2(h_conv1)

#第二层
W_conv2=weight_variable([5,5,32,64])

b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

h_pool2=max_pool_2x2(h_conv2)

#第三层全连接层

W_fc1=weight_variable([7*7*64,1024])

b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])

b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=-tf.reduce_sum(ys*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

#tf.argmax中的1表示按行取最大值
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(100)
    if i % 50 == 0:
        train_accuracy=accuracy.eval(feed_dict={xs:batch[0],ys:batch[1],keep_prob:1.0})
        print('step %d ,training accuracy %g'%(i,train_accuracy))
    train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))


'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
sess = tf.InteractiveSession

#接下来定义函数来表示数据的声明
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.Constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #第一个参数input[batch,hieght,width,in_channels],filter:卷积核,strides，滑动的步长，padding，边缘填充方式,VALID:没有填充，SAME:边缘为0.
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

def max_pool_2x2(x):
    #第一个参数input，第二个参数池化窗口的大小,一般为[1,height,width,1],strides,滑动步长，padding填充方式
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.int64,[None,10])#总共10类，手写字

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])

#接下来搭建网络，两层卷积池化层和两层全连接层

W_conv1=weight_variable([5,5,1,32])

b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积层
W_conv2=weight_variable([5,5,1,64])

b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

#第一轮图像大小变为14*14*32(一开始是28*28)，第二轮变成了7*7*64

#第三层全连接层

W_fc1 = weight_variable([7*7*64 , 1024])

b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

h_fc1_drop=tf.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])

b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(ys * tf.log(ys))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

crorect_prediction=tf.equal(tf.argmax(ys,1),tf.argmax(y_conv,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.global_variables_initializer()

for i in range(20000):
    batch=tf.nn.next_batch(50)
    if i % 100 == 0:
        train_accuracy=accuracy.eval(feed_dict = {xs:batch[0] , ys:batch[1]})
        print('train step %d train accuracy %g',%(i,train_accuracy))
    train_step.run(feed_dict={xs:batch[0],ys:batch[1]},keep_prob=0.5)
print('test accuracy:%g'%accuracy.eval(feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob=1.0}))

'''










