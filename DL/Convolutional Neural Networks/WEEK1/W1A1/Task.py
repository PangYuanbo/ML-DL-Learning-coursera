import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    # (≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    # YOUR CODE ENDS HERE

    return X_pad
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)


# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # (≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    # YOUR CODE STARTS HERE
    s=np.multiply(a_slice_prev,W)
    Z=np.sum(s)
    Z+=float(b)
    # YOUR CODE ENDS HERE

    return Z
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape (≈1 line)
    # (m, n_H_prev, n_W_prev, n_C_prev) = None
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    # Retrieve dimensions from W's shape (≈1 line)
    # (f, f, n_C_prev, n_C) = None
    f, f, n_C_prev, n_C = W.shape
    # Retrieve information from "hparameters" (≈2 lines)
    # stride = None
    # pad = None
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # n_H = None
    # n_W = None
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    # Initialize the output volume Z with zeros. (≈1 line)
    # Z = None
    Z = np.zeros((m, n_H, n_W, n_C))
    # Create A_prev_pad by padding A_prev
    # A_prev_pad = None
    A_prev_pad = zero_pad(A_prev, pad)
    # for i in range(None):               # loop over the batch of training examples
    for i in range(m):
        # a_prev_pad = None               # Select ith training example's padded activation
        a_slice_prev= A_prev_pad[i]
        # for h in range(None):           # loop over vertical axis of the output volume
        for h in range(n_H):
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            vert_start = h * stride
            vert_end = vert_start + f

            # for w in range(None):       # loop over horizontal axis of the output volume
            for w in range(n_W):
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                horiz_start = w * stride
                horiz_end = horiz_start + f


                # for c in range(None):   # loop over channels (= #filters) of the output volume
                for c in range(n_C):
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    # a_slice_prev = None
                    a_prev = a_slice_prev[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    # weights = None
                    # biases = None
                    # Z[i, h, w, c] = None
                    # YOUR CODE STARTS HERE
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_prev, weights, biases)
                    # YOUR CODE ENDS HERE

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache
np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
z_mean = np.mean(Z)
z_0_2_1 = Z[0, 2, 1]
cache_0_1_2_3 = cache_conv[0][1][2][3]
print("Z's mean =\n", z_mean)
print("Z[0,2,1] =\n", z_0_2_1)
print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
conv_forward_test_2(conv_forward)


# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    # for i in range(None):                         # loop over the training examples
    for i in range(m):
        # for h in range(None):                     # loop on the vertical axis of the output volume
        for h in range(n_H):
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            vert_start = h * stride
            vert_end = vert_start + f

            # for w in range(None):                 # loop on the horizontal axis of the output volume
            for w in range(n_W):
                # Find the vertical start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                horiz_start = w * stride
                horiz_end = horiz_start + f
                # for c in range (None):            # loop over the channels of the output volume
                for c in range(n_C):
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        # a_prev_slice = None
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        # Compute the pooling operation on the slice.
                        # Use an if statement to differentiate the modes.
                        # Use np.max and np.mean.
                        # if mode == "max":
                        # A[i, h, w, c] = None
                        if mode=="max":
                            A[i,h,w,c]=np.max(a_prev_slice)
                        # elif mode == "average":
                        # A[i, h, w, c] = None
                        elif mode=="average":
                            A[i,h,w,c]=np.mean(a_prev_slice)

    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    # assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache
# Case 1: stride of 1
print("CASE 1:\n")
np.random.seed(1)
A_prev_case_1 = np.random.randn(2, 5, 5, 3)
hparameters_case_1 = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])
A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])

pool_forward_test_1(pool_forward)

# Case 2: stride of 2
print("\n\033[0mCASE 2:\n")
np.random.seed(1)
A_prev_case_2 = np.random.randn(2, 5, 5, 3)
hparameters_case_2 = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[0] =\n", A[0])
print()

A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1] =\n", A[1])

pool_forward_test_2(pool_forward)


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    # Retrieve information from "cache"
    # (A_prev, W, b, hparameters) = None
    # Retrieve dimensions from A_prev's shape
    # (m, n_H_prev, n_W_prev, n_C_prev) = None
    # Retrieve dimensions from W's shape
    # (f, f, n_C_prev, n_C) = None
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    # Retrieve information from "hparameters"
    # stride = None
    # pad = None
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Retrieve dimensions from dZ's shape
    # (m, n_H, n_W, n_C) = None
    (m, n_H, n_W, n_C) = dZ.shape
    # Initialize dA_prev, dW, db with the correct shapes
    # dA_prev = None
    # dW = None
    # db = None
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    # Pad A_prev and dA_prev
    # A_prev_pad = zero_pad(A_prev, pad)
    # dA_prev_pad = zero_pad(dA_prev, pad)
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    # for i in range(m):                       # loop over the training examples
    for i in range(m):
        # select ith training example from A_prev_pad and dA_prev_pad
        # a_prev_pad = None
        # da_prev_pad = None
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        # for h in range(n_H):                   # loop over vertical axis of the output volume
        #    for w in range(n_W):               # loop over horizontal axis of the output volume
        #        for c in range(n_C):           # loop over the channels of the output volume
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    # vert_start = None
                    # vert_end = None
                    # horiz_start = None
                    # horiz_end = None
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the slice from a_prev_pad
                    # a_slice = None
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    # da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += None
                    # dW[:,:,:,c] += None
                    # db[:,:,:,c] += None
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
                    # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
                    # dA_prev[i, :, :, :] = None
                    # YOUR CODE STARTS HERE
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    # YOUR CODE ENDS HERE

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db
# We'll run conv_forward to initialize the 'Z' and 'cache_conv",
# which we'll use to test the conv_backward function
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 2,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

# Test conv_backward
dA, dW, db = conv_backward(Z, cache_conv)

print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))

assert type(dA) == np.ndarray, "Output must be a np.ndarray"
assert type(dW) == np.ndarray, "Output must be a np.ndarray"
assert type(db) == np.ndarray, "Output must be a np.ndarray"
assert dA.shape == (10, 4, 4, 3), f"Wrong shape for dA  {dA.shape} != (10, 4, 4, 3)"
assert dW.shape == (2, 2, 3, 8), f"Wrong shape for dW {dW.shape} != (2, 2, 3, 8)"
assert db.shape == (1, 1, 1, 8), f"Wrong shape for db {db.shape} != (1, 1, 1, 8)"
assert np.isclose(np.mean(dA), 1.4524377), "Wrong values for dA"
assert np.isclose(np.mean(dW), 1.7269914), "Wrong values for dW"
assert np.isclose(np.mean(db), 7.8392325), "Wrong values for db"

print("\033[92m All tests passed.")


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    # (≈1 line)
    # mask = None
    # YOUR CODE STARTS HERE
    mask = x == np.max(x)

    # YOUR CODE ENDS HERE
    return mask
np.random.seed(1)
x = np.random.randn(2, 3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)

x = np.array([[-1, 2, 3],
              [2, -3, 2],
              [1, 5, -2]])

y = np.array([[False, False, False],
     [False, False, False],
     [False, True, False]])
mask = create_mask_from_window(x)

assert type(mask) == np.ndarray, "Output must be a np.ndarray"
assert mask.shape == x.shape, "Input and output shapes must match"
assert np.allclose(mask, y), "Wrong output. The True value must be at position (2, 1)"

print("\033[92m All tests passed.")


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    # Retrieve dimensions from shape (≈1 line)
    # (n_H, n_W) = None
    n_H, n_W = shape
    # Compute the value to distribute on the matrix (≈1 line)
    # average = None
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value (≈1 line)
    # a = None
    # YOUR CODE STARTS HERE
    a = np.ones(shape) * average
    # YOUR CODE ENDS HERE
    return a
a = distribute_value(2, (2, 2))
print('distributed value =', a)


assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (2, 2), f"Wrong shape {a.shape} != (2, 2)"
assert np.sum(a) == 2, "Values must sum to 2"

a = distribute_value(100, (10, 10))
assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (10, 10), f"Wrong shape {a.shape} != (10, 10)"
assert np.sum(a) == 100, "Values must sum to 100"

print("\033[92m All tests passed.")


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from cache (≈1 line)
    # (A_prev, hparameters) = None
    (A_prev, hparameters) = cache
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    # stride = None
    # f = None
    stride = hparameters['stride']
    f = hparameters['f']
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    # m, n_H_prev, n_W_prev, n_C_prev = None
    # m, n_H, n_W, n_C = None
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    # Initialize dA_prev with zeros (≈1 line)
    # dA_prev = None
    dA_prev = np.zeros(A_prev.shape)
    # for i in range(None): # loop over the training examples
    for i in range(m):
        # select training example from A_prev (≈1 line)
        # a_prev = None
        a_prev = A_prev[i]
        # for h in range(n_H):                   # loop on the vertical axis
        for h in range(n_H):
            # for w in range(n_W):               # loop on the horizontal axis
            for w in range(n_W):
                # for c in range(n_C):           # loop over the channels (depth)
                for c in range(n_C):

                # Find the corners of the current "slice" (≈4 lines)
                # vert_start = None
                # vert_end = None
                # horiz_start = None
                # horiz_end = None
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Compute the backward propagation in both modes.
                    # if mode == "max":
                    if mode== "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        # a_prev_slice = None
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        # mask = None
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

                    # elif mode == "average":
                    elif mode == "average":

                        # Get the value da from dA (≈1 line)
                        # da = None
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        # shape = None
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
                        # YOUR CODE STARTS HERE
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

                        # YOUR CODE ENDS HERE

                # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


# GRADED FUNCTION: forward_propagation_with_dropout

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # (≈ 4 lines of code)         # Steps 1-4 below correspond to the Steps 1-4 described above.
    # D1 =                                           # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    # D1 =                                           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    # A1 =                                           # Step 3: shut down some neurons of A1
    # A1 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # YOUR CODE ENDS HERE
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    # (≈ 4 lines of code)
    # D2 =                                           # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    # D2 =                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    # A2 =                                           # Step 3: shut down some neurons of A2
    # A2 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache