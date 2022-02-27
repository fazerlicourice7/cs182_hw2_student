from multiprocessing import pool
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # print(f"shape of x: {np.shape(x)}")
    input = np.reshape(x, (np.shape(x)[0], -1))
    # print(f"shape of input: {np.shape(input)}")
    out = np.matmul(input, w) + b
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    dx = np.matmul(dout, w.T).reshape(np.shape(x)) # need to reshape this.
    N = np.shape(x)[0]
    x_ = np.reshape(x, (N, -1))
    dw = np.matmul(x_.T, dout)
    db = np.matmul(dout.T, np.ones(N))
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x,0)
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    relu_deriv = lambda x: 1 if x > 0 else 0
    vrelu_deriv = np.vectorize(relu_deriv)
    deriv = vrelu_deriv(x)
    dx = np.multiply(dout, deriv)
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        mu = np.mean(x, axis=0)
        # print(f"shape of mu: {np.shape(mu)}, should be equal to second dimension in shape of x: {np.shape(x)}")
        var = np.var(x, axis=0)
        sigma = np.sqrt(var + eps) # + np.sqrt(eps)
        # print(f"shape of sigma: {np.shape(sigma)}, should be equal to second dimension in shape of x: {np.shape(x)}")
        out = (x - mu) / (sigma)
        cache={'gamma':gamma, 'z':out, 'mu':mu, 'sigma':sigma, 'var': var, 'x':x}

        out = np.multiply(out, gamma)
        out = np.add(out, beta)

        running_mean = momentum * running_mean + (1-momentum) * mu
        running_var  = momentum * running_var  + (1-momentum) * var

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        out = (x - running_mean) / np.sqrt(running_var + eps)
        out = np.multiply(out, gamma)
        out = np.add(out, beta)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    mu = cache['mu']
    x = cache['x']
    z = cache['z']
    var = cache['var']
    std = cache['sigma']
    gamma = cache['gamma']

    N = np.shape(dout)[0]

    dgamma = np.sum(dout * z, axis=0)
    dbeta = np.sum(dout, axis=0)

    #z derivatives
    dz_dx = 1.0/(std)
    dz_dmu = -1.0/std
    dz_dstd = -1.0*(x-mu)/(std**2)

    #Mean derivative
    dmu_dx = 1.0 / N

    #Std derivatives
    dstd_dvar = 0.5 * (var**-0.5)
    dvar_dx1 = 2.0/N * (x-mu)
    dvar_dmu = 2.0/N * np.sum(x-mu,axis=0) * -1.0
    dvar_dx2 = dvar_dmu * dmu_dx


    dx = dout*gamma*dz_dx + \
     np.sum(dout*gamma*dz_dmu,axis=0)*dmu_dx +\
     np.sum(dout*gamma*dz_dstd,axis=0)*dstd_dvar*(dvar_dx1 + dvar_dx2)

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    mu = cache['mu']
    x = cache['x']
    z = cache['z']
    var = cache['var']
    std = cache['sigma']
    gamma = cache['gamma']

    N = np.shape(dout)[0]

    dgamma = np.sum(dout * z, axis=0)
    dbeta = np.sum(dout, axis=0)

    # dout/dz
    dz = dout * gamma

    dx = (N * dz - np.sum(dz, axis=0) - z * np.sum(z * dz, axis=0)) / (N * std)

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask = np.random.choice([0,1],p=[p,1-p], size=np.shape(x))
        out =(mask * x) * 1/(1-p)
        pass
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = dout * mask
        pass
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    padded_x = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))
    # print(f"shape of x: {np.shape(x)}, shape of padded_x: {np.shape(padded_x)}")
    F = np.shape(w)[0]
    HH = np.shape(w)[2]
    WW = np.shape(w)[3]

    N = np.shape(x)[0]
    C = np.shape(x)[1]
    H = np.shape(x)[2]
    W = np.shape(x)[3]

    # print(f"N, C, H, W: {N}, {C}, {H}, {W}")
    # print(f"F, C, HH, WW: {F}, {C}, {HH}, {WW}")

    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)
    # print(f"output sizes: H': {H_prime}, W': {W_prime}")

    out = np.empty((N,F,H_prime,W_prime),dtype=x.dtype)

    for n in range(N):
      for f in range(F):
        for _h in range(H_prime):
          for _w in range(W_prime):
            h_start = _h * stride
            w_start = _w * stride

            h_end = h_start + HH
            w_end = w_start + WW

            out[n,f,_h,_w] = np.sum(padded_x[n,:,h_start:h_end,w_start:w_end] * w[f,:,:,:]) + b[f]

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param = cache

    pad = conv_param['pad']
    stride = conv_param['stride']

    padded_x = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))

    dw = np.zeros(np.shape(w))
    dx = np.zeros(np.shape(padded_x))

    F = np.shape(w)[0]
    HH = np.shape(w)[2]
    WW = np.shape(w)[3]

    N = np.shape(x)[0]
    C = np.shape(x)[1]
    H = np.shape(x)[2]
    W = np.shape(x)[3]

    H_prime = np.shape(dout)[2]
    W_prime = np.shape(dout)[3]

    print(f"shapes:\nw:{np.shape(w)}\nx:{np.shape(x)}\npadded_x:{np.shape(padded_x)}\ndout:{np.shape(dout)}")

    db = np.sum(dout, (0,2,3))
    for n in range(N):
      for f in range(F):
        for _h in range(H_prime):
          for _w in range(W_prime):
            h_start = _h * stride
            w_start = _w * stride

            h_end = h_start + HH
            w_end = w_start + WW

            dw[f,:,:,:] += padded_x[n,:,h_start:h_end,w_start:w_end] * dout[n,f,_h,_w]
            dx[n,:,h_start:h_end, w_start:w_end] += w[f,:,:,:] * dout[n,f,_h,_w]

            #out[n,f,_h,_w] = np.sum(padded_x[n,:,h_start:h_end,w_start:w_end] * w[f,:,:,:]) + b[f]

    dx = dx[:,:,pad:-pad, pad:-pad]

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = np.shape(x)
    mask = np.zeros(np.shape(x))

    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = int(1 + (H  - HH) / stride)
    W_prime = int(1 + (W  - WW) / stride)

    out = np.zeros((N,C,H_prime,W_prime),dtype=x.dtype)

    for i in range(H_prime):
      for j in range(W_prime):
        h_i = i*stride
        h_f = h_i + H_prime
        w_i = j*stride
        w_f = w_i + W_prime

        pool_region = x[:,:,h_i:h_f,w_i:w_f]
        val = np.max(pool_region, axis=(2,3))
        # print(f"shape of max: {np.shape(val)}")
        # idx = np.argmax(pool_region, axis=(2,3))
        # idx = np.argmax(pool_region, axis=(2,3))
        # print(f"argmax: {idx}")
        print(f"shape of pool_region: {np.shape(pool_region)}")
        # idx_rc = np.unravel_index(8, np.array(pool_region).shape[:2])
        # print(f"idx_rc: {idx_rc}")

        out[:,:,i,j] = val

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    x = np.asarray(x)
    dx = np.zeros(np.shape(x))

    N, C, H, W = np.shape(x)
    print(f"shape of x: {np.shape(x)}")

    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = int(1 + (H  - HH) / stride)
    W_prime = int(1 + (W  - WW) / stride)
    print(f"H': {H_prime}, W': {W_prime}")

    for i in range(H_prime):
      for j in range(W_prime):
        print(f"({i}, {j})")
        h_i = i*stride
        h_f = h_i + H_prime
        w_i = j*stride
        w_f = w_i + W_prime

        print(f"h_i={h_i}, h_f={h_f}, w_i={w_i}, w_f={w_f}")
        pool_region = x[:,:,h_i:h_f,w_i:w_f]
        val = np.max(pool_region, axis=(2,3))

        # print(f"poo region: {pool_region}")
        # print(f"max vals: {val}")

        mask = np.zeros(np.shape(pool_region))

        for n in range(N):
          for c in range(C):
            idx = np.argwhere(pool_region[n, c] == val[n,c])
            print(f"npargwhere: {idx}")
            for ix in idx:
              print(f"i: {ix}")
              mask[n, c, ix[0], ix[1]] = dout[n, c, i, j]
        
        print(f"h_i={h_i}, h_f={h_f}, w_i={w_i}, w_f={w_f}")
        print(f"shape of mask: {np.shape(mask)}, shape of dx subset: {np.shape(dx[:,:,h_i:h_f, w_i:w_f])}")
        dx[:,:,h_i:h_f, w_i:w_f] = mask
    # dx *= dout

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    # mode = bn_param['mode']
    # eps = bn_param.get('eps', 1e-5)
    # momentum = bn_param.get('momentum', 0.9)

    N, C, H, W = np.shape(x)
    # # running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    # # running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    # mu = x.mean(axis=(0, 2, 3))
    # var = np.var(x, axis=(0, 2, 3))
    # sigma = np.sqrt(var) + np.sqrt(eps)
    # print(f"shape of mu: {np.shape(mu)}, shape of var: {np.shape(var)}")
    # out = (x - mu) / (sigma)
    # cache={'gamma':gamma, 'z':out, 'mu':mu, 'sigma':sigma, 'var': var, 'x':x}
    # print(f"shape of out before scale, shift: {np.shape(out)}")
    # out = np.multiply(out, gamma)
    # out = np.add(out, beta)
    # print(f"shape of out after scale, shift: {np.shape(out)}")

    # running_mean = momentum * running_mean + (1-momentum) * mu
    # running_var  = momentum * running_var  + (1-momentum) * var

    # print(f"shape of gamma, beta: {np.shape(gamma)}, {np.shape(beta)}")
    reshaped_x = np.reshape(x, (-1, C))
    # print(f"shape of reshaped x: {np.shape(reshaped_x)}")
    interim, cache = batchnorm_forward(reshaped_x, gamma, beta, bn_param)
    # print(f"shape of running mean, var: {np.shape(bn_param['running_mean'])}, {np.shape(bn_param['running_var'])}")
    # print(f"shape of interim: {np.shape(interim)}")

    out = np.reshape(interim, (N, C, H, W))
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = np.shape(dout)
    reshaped_dout = np.reshape(dout, (-1, C))
    # print(f"shape of reshaped x: {np.shape(reshaped_x)}")
    dx, dgamma, dbeta = batchnorm_backward(reshaped_dout, cache)
    # print(f"shape of running mean, var: {np.shape(bn_param['running_mean'])}, {np.shape(bn_param['running_var'])}")
    # print(f"shape of interim: {np.shape(interim)}")
    # print(f"shape of dx: {np.shape(dx)}, shape of dgamma: {np.shape(dgamma)}, shape of dbeta: {np.shape(dbeta)}")

    dx = np.reshape(dx, (N, C, H, W))
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
