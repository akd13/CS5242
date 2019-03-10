import numpy as np
from utils.tools import img2col,col2img

# Attension:
# - Never change the value of input, which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operation):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operation):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class fc(operation):
    def __init__(self):
        super(fc, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of fc layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


class conv(operation):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad = 2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad'] # padding on each side
        stride = self.conv_params['stride'] 
        in_channel = self.conv_params['in_channel'] # rgb
        out_channel = self.conv_params['out_channel'] # number of filters
        
        output = None
        
        #########################################  
        batch,in_height,in_width = input.shape[0],input.shape[2],input.shape[3]
        out_height = int(np.floor((in_height+2*pad-kernel_h)/stride)+1)
        out_width = int(np.floor((in_width+2*pad-kernel_w)/stride)+1)
        input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        in_height,in_width = input_pad.shape[2],input_pad.shape[3]
        
        h_indices = np.arange(0,in_height-kernel_h+1,stride) # the height indices for receptive fields
        w_indices = np.arange(0,in_width-kernel_w+1,stride) # the width indices for receptive fields
        
        receptive_fields = img2col(input_pad,h_indices,w_indices,kernel_h, kernel_w) # get all receptive fields in batch
        kernel_matrix = np.reshape(weights,(out_channel,in_channel*kernel_h*kernel_w)) 
        bias_reshape = np.repeat(bias, out_height*out_width,axis=0).reshape(out_channel,out_height*out_width)        
        output = np.vstack(list(map(lambda x: np.matmul(kernel_matrix,x)+bias_reshape, receptive_fields))) # y = wx+b   
        output = output.reshape(batch, out_channel, out_height, out_width)
        #########################################  

        return output
    
    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']
        
        #########################################
        #initialize shapes for in_grad, w_grad
        
        in_grad = np.zeros(input.shape)
        w_grad = np.zeros(weights.shape)
        
        batch,in_height,in_width = input.shape[0],input.shape[2],input.shape[3]
        out_height,out_width = out_grad.shape[2],out_grad.shape[3]
        out_height = int(np.floor((in_height+2*pad-kernel_h)/stride)+1)
        out_width = int(np.floor((in_width+2*pad-kernel_w)/stride)+1)
        input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        in_height,in_width = input_pad.shape[2],input_pad.shape[3]

        # calculate weight gradient, w_grad, i.e. dW          
        h_indices_dW = np.arange(0,in_height-kernel_h+1,stride) # the height indices for receptive fields
        w_indices_dW = np.arange(0,in_width-kernel_w+1,stride) # the width indices for receptive fields
        
        #TODO: np.vtack, map, lambda        
        for b in range(batch):
            cur_input = input_pad[b].reshape(1,in_channel,in_height, in_width)
            cur_out = out_grad[b].reshape(out_channel, out_height*out_width)   
            transformed_input = img2col(cur_input,h_indices_dW,w_indices_dW,kernel_h,kernel_w)
            transformed_input = transformed_input.reshape(in_channel*kernel_h*kernel_w,out_height*out_width)
            temp_product = np.matmul(cur_out,transformed_input.T).reshape(out_channel, in_channel, kernel_h, kernel_w)   
            w_grad+=temp_product.reshape(out_channel, in_channel, kernel_h, kernel_w)   
        
        # Calculate input gradient to forward layer, i.e. dX
        weight_matrix = np.reshape(weights,(out_channel,in_channel*kernel_h*kernel_w))
        
        #TODO: np.vtack, map, lambda
        for b in range(batch):                
            cur_out = out_grad[b].reshape(out_channel,out_height*out_width)
            col2img_input = np.matmul(weight_matrix.T,cur_out) 
            in_grad[b] = col2img(in_grad[b], col2img_input,out_height,out_width,in_channel,kernel_h,kernel_w,stride)
        
        # get bias by summing gradient along axis 1
        b_grad = np.sum(out_grad, axis=(0, 2, 3))

        #########################################

        return in_grad, w_grad, b_grad
        

class pool(operation):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad = 2 means a 2-pixel border of padding with zeros.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        output = None

        #########################################
        
        batch,in_channel,in_height,in_width = input.shape
        out_height = int(np.floor((in_height+2*pad-pool_height)/stride)+1)
        out_width = int(np.floor((in_width+2*pad-pool_width)/stride)+1)
        input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        in_height,in_width = input_pad.shape[2],input_pad.shape[3]
    
        output = np.zeros((batch, in_channel, out_height, out_width))

        h_indices = np.arange(0,in_height-pool_height+1,stride) # the height indices for receptive fields
        w_indices = np.arange(0,in_width-pool_width+1,stride) # the width indices for receptive fields
        
        receptive_fields = img2col(input_pad,h_indices,w_indices,pool_height, pool_width) # get all receptive fields in batch
        
        #TODO: np.vtack, map, lambda        
        for b in range(batch):
            current_image = receptive_fields[b]
            batch_pool = []
            # slice current image into c slices for pooling
            sliced_list = list(map(lambda cur,c: cur[c*pool_height*pool_width:(c+1)*pool_height*pool_width,],[current_image]*in_channel,range(in_channel)))
            
            if(pool_type == 'max'):
                batch_pool = list(map(lambda sliced:sliced[np.argmax(sliced, axis=0), range(np.argmax(sliced, axis=0).size)].reshape(1,out_height*out_width),sliced_list))
            elif(pool_type == 'avg'):
                batch_pool = list(map(lambda sliced:np.mean(sliced,axis=0).reshape(1,out_height*out_width),sliced_list))
            else:
                raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
            output[b] = np.vstack(tuple(batch_pool)).reshape(in_channel,out_height,out_width)
        
        #########################################
        
        return output
    
    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        
        # refer: https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/
        # due to sparse/repeated entries during matrix multiplication for max/avg pooling, we use loops and slicing instead.
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch,in_channel,in_height,in_width = input.shape
        out_height = int(np.floor((in_height+2*pad-pool_height)/stride)+1)
        out_width = int(np.floor((in_width+2*pad-pool_width)/stride)+1)        
        input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        in_height,in_width = input_pad.shape[2],input_pad.shape[3]
        
        in_grad = np.zeros((batch, in_channel, in_height, in_width))
        
        #TODO: np.vtack, map, lambda        
        for b in range(batch):
            current_image = input_pad[b]
            for c in range(in_channel):
                for h in range(out_height):
                    for w in range(out_width):
                        temp_slice = current_image[c, h*stride: h*stride+pool_height, w*stride: w*stride+pool_width]
                        if(pool_type=='max'):
                            max_out = np.zeros(temp_slice.shape)
                            max_out[np.unravel_index(np.argmax(temp_slice, axis=None), temp_slice.shape)]=1
                            in_grad[b,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width]+=max_out*out_grad[b,c,h,w]
                        elif(pool_type=='avg'):
                            avg_out = np.ones(temp_slice.shape)*(1/(pool_height*pool_width))
                            in_grad[b,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width]+=avg_out*out_grad[b,c,h,w]
                        else:
                            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        #########################################
        
        return in_grad

class dropout(operation):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        output = None
        if self.training:
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            #########################################
            if self.mask is None:
                self.mask = np.random.binomial(1,self.rate,input.shape)
            input*=self.mask
            output = input/(1-self.rate)
            #########################################
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            #########################################
            in_grad = (out_grad*self.mask)/(1-self.rate)
            #########################################
        else:
            in_grad = out_grad
        return in_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
