import numpy as np
import copy
np.random.seed(2018)

#激活函数
def sigmoid(x):
    output=1/(1+np.exp(-x))
    return output

#激活函数的求导
def sigmoid_output_to_derivative(output):
    return output*(1-output)

#生成数据集
int2binary={}
binary_dim=8#设置二进制数据的长度，相当于限制了数据的取值大小

largest_number=pow(2,binary_dim)
binary=np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T, axis=1)#将数字转换成二进制

for i in range(largest_number):
    int2binary[i]=binary[i]

alpha=0.1
input_dim=2
hidden_dim=16
output_dim=1

#设置权重矩阵的大小
synapse_0=2*np.random.random((input_dim,hidden_dim))-1
synapse_1=2*np.random.random((hidden_dim,output_dim))-1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim))-1

synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)


for j in range(10000):
    a_int=np.random.randint(largest_number/2)
    a=int2binary[a_int]

    b_int=np.random.randint(largest_number/2)
    b=int2binary[b_int]

    c_int=a_int+b_int
    c=int2binary[c_int]

    d=np.zeros_like(c)

    overallError=0

    layer_2_deltas=list()
    layer_1_values=list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        X=np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])
        y=np.array([[c[binary_dim-position-1]]]).T

        #hidden layer(input +pre_hidden)
        #layer_1_values[-1]取最后一个保存的隐藏层的数，也就是上一个数
        layer_1=sigmoid(np.dot(X,synapse_0)+np.dot(layer_1_values[-1],synapse_h))

        layer_2=sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error=y-layer_2#78
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))#误差函数求导*softmax求导
        overallError+=np.abs(layer_2_error[0])

        d[binary_dim-position-1]=np.round(layer_2[0][0])#83

        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta=np.zeros(hidden_dim)

        #接下来就是反向传播的代码了
    for position in range(binary_dim):

        X=np.array([[a[position],b[position]]])
        layer_1=layer_1_values[-position-1]#保存当前状态的值
        prev_layer_1=layer_1_values[-position-2]#保存前一状态的值
        
        layer_2_delta=layer_2_deltas[-position-1]
        
        layer_1_delta=(future_layer_1_delta.dot(synapse_h.T)+layer_2_delta.dot(
            synapse_1.T))*sigmoid_output_to_derivative(layer_1)
        
        synapse_1_update+=np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update+=np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update+=X.T.dot(layer_1_delta)
        '''
        对于隐藏层到输出层的权重矩阵V,其中z=sigmoid(V*h_t)
        dl/dV=dl/dy_t*dy_t/dz_t*dz_t/dV=dl/dy_t*dy_t/dz_t*h_t
        layer_1==>h_t
        dl/dy_t*dy_t/dz_t==>layer_2_delta


        这里面用d来代替偏导数,l是误差,l_t是t时刻的误差,W是h_t-1到h_t的权重函数
        S=U*x_t+W*h_t-1
        dl_t/dW=dl_t/dy_t*dy_t/dz*dz/dh_t*dh_t/dS*dS/dW,
        layer_2_delta.dot(synapse_1.T)===>>>dl_t/dy_t*dy_t/dz*dz/dh_t
        dh_t/dS===>>sigmoid_output_to_derivative(layer_1)
    
        dl/dU=dl/dy_t*dy_t/dz_t*dz_t/dh_t*dh_t/dS*dS/dV
        layer_1_delta===>>dl/dy_t*dy_t/dz_t*dz_t/dh_t*dh_t/dS
        dS/dV===>>X.T
        
        不管上面的叙述和代码，上面代码激活函数都是sigmoid
        s_t=tanh(U*x_t+W*s_t-1)
        y=softmax(V*s_t),其中y1=V*s_t
        dl_t/dW=dl_t/dy1*dy1/ds_t*ds_t/dW利用链式法则等价于
        =sum(from 0 to t)dl_t/dy1*dy1/ds_t*ds_t/ds_k*ds_k/dW
        第一次k=t,也就是取消了ds_t/ds_k这一项，前面的结果dl_t/dy1*dy1/ds_t====>>layer_2_delta
        这一块没搞懂后面为什么还要加上layer_2_delta.dot(****)
        '''
        

        future_layer_1_delta=layer_1_delta

    synapse_0+=synapse_0_update*alpha
    synapse_1+=synapse_1_update*alpha
    synapse_h+=synapse_h_update*alpha

    synapse_0_update*=0
    synapse_1_update*=0
    synapse_h_update*=0

    if (j%1000 == 0):
        print('Error:%s'%str(overallError))
        print('Pred:%s'%str(d))
        print('True:%s'%str(c))
        out=0
        for index,x in enumerate(reversed(d)):
            out+=x*pow(2,index)
        print(str(a_int)+'+'+str(b_int)+'='+str(out))
        print('----------------------------')

'''


import copy, numpy as np
np.random.seed(0)
 
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
 
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
 
# training dataset generation
int2binary = {}
binary_dim = 8
 
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
 
 
# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
 
 
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
 
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
 
# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding
 
    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding
 
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)
 
    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
 
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
 
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
 
        # did we miss?... if so by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
            layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
 
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    
 
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print("Error:",str(overallError))
        print("Pred:",str(d))
        print("True:",str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
'''






