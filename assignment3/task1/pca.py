import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




# Normalize each row to one (w / sum of all w's in row)
def normalizeWeigths(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

def paddingImage(image_data, image_size):
    new_size = (image_size, image_size)
    pad_image = np.zeros(new_size)
    pad_image[:image_data.shape[0],:image_data.shape[1]] = image_data
    return pad_image

# original_image_size is a tuple of original_image_data dimension
def unpaddingImage(image_data, original_image_size):
    unpad_image = np.zeros(original_image_size)
    unpad_image = image_data[:original_image_size[0],:original_image_size[1]]
    return unpad_image

# Calculate the Eigen values
def train(epoch, image_data):
    global weightMatrices
    global eigenValues
    global number_of_submatrices
    for i_submatrix in range(number_of_submatrices):
        for j_submatrix in range(number_of_submatrices):
            submatrix = image_data[i_submatrix * MASK_SIZE : i_submatrix * MASK_SIZE + MASK_SIZE, 
                                   j_submatrix * MASK_SIZE : j_submatrix * MASK_SIZE + MASK_SIZE].flatten().astype('float32')
            
            #weights = np.random.rand(NUM_COMPONENTS, MASK_SIZE*MASK_SIZE)
            #weights = normalizeWeigths(weights)
            weights = weightMatrices[i_submatrix][j_submatrix]
            weights = normalizeWeigths(weights)
            y = np.zeros((NUM_COMPONENTS,)).astype('float32')
            for j in range(NUM_COMPONENTS):
                w_j = np.reshape(weights[j, :], (1, MASK_SIZE*MASK_SIZE))
                y_j = w_j.dot(submatrix)

                y[j] = y_j
                
                #Calculate delta weight
                x_prime = submatrix
                for k in range(1,j):
                    x_prime -= weights[k, :]*y[k]
                delta_w_j = LEARNING_RATE * y_j*(x_prime - y_j*weights[j])

                # update weight
                weights[j] = weights[j] + delta_w_j
            
            weights = normalizeWeigths(weights)
            # Update weightMatrix and eigenMatrix (output)
            weightMatrices[i_submatrix][j_submatrix] = weights
            eigenValues[i_submatrix][j_submatrix] = y      
            


if __name__ == '__main__':
    image = Image.open('./images/Lena.jpg').convert('L')

    original_image_data = np.asarray(image) / 255
    print("Original Image Size")
    print(original_image_data.shape)
    #plt.imshow(original_image_data, cmap='gray')
    #plt.title('Original Image')
    #plt.savefig('./original_tropical_tree.png')
    #plt.show()

    LEARNING_RATE = 1e-4
    MASK_SIZE = 4 
    EPOCHS = 2000
    NUM_COMPONENTS = 8

    ORIGINAL_IMAGE_SIZE = max(original_image_data.shape)
    if ORIGINAL_IMAGE_SIZE % MASK_SIZE != 0:
        IMAGE_SIZE = ORIGINAL_IMAGE_SIZE + MASK_SIZE - (ORIGINAL_IMAGE_SIZE% MASK_SIZE)
        is_padding = True
    else:
        IMAGE_SIZE = ORIGINAL_IMAGE_SIZE
        is_padding = False
    
    number_of_submatrices = int(IMAGE_SIZE / MASK_SIZE)
    
    if is_padding:
        image_data = paddingImage(original_image_data, IMAGE_SIZE)
    else:
        image_data = original_image_data
    
    
    weightMatrices = np.random.rand(number_of_submatrices, number_of_submatrices, NUM_COMPONENTS, MASK_SIZE*MASK_SIZE).astype('float32')
    weightMatrices = normalizeWeigths(weightMatrices)
    eigenValues = np.zeros((number_of_submatrices, number_of_submatrices, NUM_COMPONENTS))
    for epoch in range(EPOCHS):
        train(epoch,  image_data)
    
    # Reconstruct the image
    reconstructed_image = np.zeros(image_data.shape)
    for i_submatrix in range(number_of_submatrices):
        for j_submatrix in range(number_of_submatrices):
            
            submatrix = (weightMatrices[i_submatrix][j_submatrix].T).dot(eigenValues[i_submatrix][j_submatrix])
            submatrix = np.reshape(submatrix, (MASK_SIZE, MASK_SIZE))
            reconstructed_image[i_submatrix * MASK_SIZE : i_submatrix * MASK_SIZE + MASK_SIZE,
                                j_submatrix * MASK_SIZE : j_submatrix * MASK_SIZE + MASK_SIZE] = submatrix
    
    if is_padding:
        reconstructed_image = unpaddingImage(reconstructed_image, original_image_data.shape)
    
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.savefig('reconstructed_Lena_mask4.png')
    #plt.show()
