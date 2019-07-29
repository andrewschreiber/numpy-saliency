import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import os
import copy


def save_vanilla_gradient(network, data, labels, target_class):
    """ Implements gradient visualization with vanilla backprop. """

    # Extract image and labels from data that match target_class
    target_indexes = [i for i in range(len(labels))
                      if np.argmax(labels[i]) == target_class]
    label_one_hot = labels[target_indexes[0]]
    target_images = [data[index] for index in target_indexes]
    for i, digit_img in enumerate(target_images):
        # Put input into layers
        output = digit_img
        for l in range(len(network.layers)):
            output = network.layers[l].forward(output)

        # Backprop to get gradient
        dy = np.array(label_one_hot)
        for l in range(len(network.layers)-1, -1, -1):
            dout = network.layers[l].backward(dy)
            dy = dout

        # Remove color channel, zero padding
        vertical_trimmed = dout[0][2:30]
        horizontal_trimmed = []
        for row in vertical_trimmed:
            horizontal_trimmed.append(row[2:30])
        trimmed = np.array(horizontal_trimmed)

        im = normalize_array(digit_img)
        im = format_np_output(im)
        im = Image.fromarray(im)
        norm_gradient = normalize_array(trimmed)

        save_gradient_overlay_images(im, norm_gradient, "index-" + str(
            i) + "_class-" + str(target_class) + '_vanilla')

        print("Saved Vanilla Gradient image to results folder")


def normalize_array(arr):
    """
        Sets array values to span 0-1
    """
    arr = arr - arr.min()
    arr /= arr.max()
    return arr


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Normalize
    gradient = normalize_array(gradient)
    # Save image
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_gradient_overlay_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')

    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(
        org_img, activation_map, 'RdBu')

    # Save original
    path_to_file = os.path.join('results', file_name+'_base.png')
    save_image(org_img, path_to_file)

    # Save heatmap on image
    path_to_file = os.path.join('results', file_name+'_saliency_overlay.png')
    save_image(heatmap_on_image, path_to_file)

    # Save colored heatmap
    path_to_file = os.path.join('results', file_name+'_saliency.png')
    save_image(heatmap, path_to_file)

    # Save grayscale heatmap
    # path_to_file = os.path.join('results', file_name+'_grayscale.png')
    # save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image
