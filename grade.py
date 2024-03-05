from PIL import Image
from PIL import ImageFilter, ImageOps, ImageDraw, ImageChops

# This is only used for the progress bar output, as I find this useful feedback when writing scripts that may take a while
from tqdm import tqdm

import numpy as np
import sys

# Change back-end to avoid headless errors
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Points around the questions on the blank form data
# These were manually labeled by Paul and are given as:
# key:value pairs where question number is the key and the coordinates [left, bottom, right_number, top, right_full_box] are the value
# Like this: question_number:[left, bottom, right_number, top, right_full_box]
# right_number is only the digit we are looking for and not the following boxes
# right_full_box is the rightmost point of all 5 fill-in boxes
pre_det_box_loc_blank_form = {
    1:[198, 696, 246, 654, 529],
    2:[198, 745, 246, 703, 529],
    3:[198, 795, 246, 753, 529],
    4:[198, 844, 246, 803, 529],
    5:[198, 894, 246, 852, 529],
    6:[198, 943, 246, 902, 529],
    7:[198, 993, 246, 951, 529],
    8:[198, 1043, 246, 1001, 529],
    9:[198, 1092, 246, 1050, 529],
    10:[198, 1142, 246, 1100, 529],
    11:[198, 1191, 246, 1150, 529],
    12:[198, 1241, 246, 1199, 529],
    13:[198, 1290, 246, 1249, 529],
    14:[198, 1340, 246, 1298, 529],
    15:[198, 1390, 246, 1348, 529],
    16:[198, 1439, 246, 1397, 529],
    17:[198, 1489, 246, 1447, 529],
    18:[198, 1538, 246, 1496, 529],
    19:[198, 1588, 246, 1546, 529],
    20:[198, 1637, 246, 1596, 529],
    21:[198, 1687, 246, 1645, 529],
    22:[198, 1736, 246, 1695, 529],
    23:[198, 1786, 246, 1744, 529],
    24:[198, 1836, 246, 1794, 529],
    25:[198, 1885, 246, 1843, 529],
    26:[198, 1935, 246, 1893, 529],
    27:[198, 1984, 246, 1943, 529],
    28:[198, 2034, 246, 1992, 529],
    29:[198, 2083, 246, 2042, 529],

    30:[646, 696, 695, 654, 978],
    31:[646, 745, 695, 703, 978],
    32:[646, 795, 695, 753, 978],
    33:[646, 844, 695, 803, 978],
    34:[646, 894, 695, 852, 978],
    35:[646, 943, 695, 902, 978],
    36:[646, 993, 695, 951, 978],
    37:[646, 1043, 695, 1001, 978],
    38:[646, 1092, 695, 1050, 978],
    39:[646, 1142, 695, 1100, 978],
    40:[646, 1191, 695, 1150, 978],
    41:[646, 1241, 695, 1199, 978],
    42:[646, 1290, 695, 1249, 978],
    43:[646, 1340, 695, 1298, 978],
    44:[646, 1390, 695, 1348, 978],
    45:[646, 1439, 695, 1397, 978],
    46:[646, 1489, 695, 1447, 978],
    47:[646, 1538, 695, 1496, 978],
    48:[646, 1588, 695, 1546, 978],
    49:[646, 1637, 695, 1596, 978],
    50:[646, 1687, 695, 1645, 978],
    51:[646, 1736, 695, 1695, 978],
    52:[646, 1786, 695, 1744, 978],
    53:[646, 1836, 695, 1794, 978],
    54:[646, 1885, 695, 1843, 978],
    55:[646, 1935, 695, 1893, 978],
    56:[646, 1984, 695, 1943, 978],
    57:[646, 2034, 695, 1992, 978],
    58:[646, 2083, 695, 2042, 978],

    59:[1096, 696, 1143, 654, 1427],
    60:[1096, 745, 1143, 703, 1427],
    61:[1096, 795, 1143, 753, 1427],
    62:[1096, 844, 1143, 803, 1427],
    63:[1096, 894, 1143, 852, 1427],
    64:[1096, 943, 1143, 902, 1427],
    65:[1096, 993, 1143, 951, 1427],
    66:[1096, 1043, 1143, 1001, 1427],
    67:[1096, 1092, 1143, 1050, 1427],
    68:[1096, 1142, 1143, 1100, 1427],
    69:[1096, 1191, 1143, 1150, 1427],
    70:[1096, 1241, 1143, 1199, 1427],
    71:[1096, 1290, 1143, 1249, 1427],
    72:[1096, 1340, 1143, 1298, 1427],
    73:[1096, 1390, 1143, 1348, 1427],
    74:[1096, 1439, 1143, 1397, 1427],
    75:[1096, 1489, 1143, 1447, 1427],
    76:[1096, 1538, 1143, 1496, 1427],
    77:[1096, 1588, 1143, 1546, 1427],
    78:[1096, 1637, 1143, 1596, 1427],
    79:[1096, 1687, 1143, 1645, 1427],
    80:[1096, 1736, 1143, 1695, 1427],
    81:[1096, 1786, 1143, 1744, 1427],
    82:[1096, 1836, 1143, 1794, 1427],
    83:[1096, 1885, 1143, 1843, 1427],
    84:[1096, 1935, 1143, 1893, 1427],
    85:[1096, 1984, 1143, 1943, 1427]
}


def convolve_with_fft(input_image, kernel, kernel_size=3, same_shape=False):
    # FFT code taken from 4.9 and converted to work on the input image (a numpy array in this function):
    #fft2 = np.fft.fftshift(np.fft.fft2(np.asarray(output_im)))
    # Display the FFT space:
    #output_im = Image.fromarray((np.log(abs(fft2))* 255 /np.amax(np.log(abs(fft2)))).astype(np.uint8), mode="L")
    # Convert back to image:
    #output_im = Image.fromarray((abs(np.fft.ifft2(np.fft.ifftshift(fft2)))).astype(np.uint8), mode="L")

    # Take the fft of the input image:
    fft2_input_image = np.fft.fftshift(np.fft.fft2(input_image))
    # Pad the kernel to be the same size as the input image:
    if (same_shape == False):
        fft2_kernel = np.fft.fftshift(np.fft.fft2(np.pad(np.reshape(np.array(kernel), (kernel_size, kernel_size)), ((0, input_image.shape[0] - kernel_size), (0, input_image.shape[1] - kernel_size)), mode="constant")))
    else:
        # Added this just in case we want to convolve small areas
        fft2_kernel = np.fft.fftshift(np.fft.fft2(kernel))

    # Perform the convolution in frequency space:
    fft2_kernel_conv = fft2_input_image * fft2_kernel

    # Convert back to spacial domain:
    applied_conv = abs(np.fft.ifft2(np.fft.ifftshift(fft2_kernel_conv)))

    return applied_conv


def sobel_filter(input_im, window_size=3):
    #print(f'Image size (width, height): ({form_image.width}, {form_image.height})')

    # Scale the bluring based on the window size used
    input_im = input_im.filter(ImageFilter.MinFilter(window_size))
    input_im = input_im.filter(ImageFilter.GaussianBlur(radius=np.floor(window_size / 2)))
    
    sobel_x = [
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    ]

    sobel_y = [
        1,  2,  1,
        0,  0,  0,
       -1, -2, -1,
    ]

    # Use an fft to calculate the applied kernels more details
    # Had to calculate the sobel filters using an fft due to issues with ImageFilter truncating negative values in uint8 images
    input_im = np.asarray(input_im)
    sobel_x = convolve_with_fft(input_im, sobel_x, 3) / 8
    sobel_y = convolve_with_fft(input_im, sobel_y, 3) / 8

    # Shift the filter back
    sobel_x = np.roll(sobel_x, -int((window_size - 1) / 2), axis=0)
    sobel_x = np.roll(sobel_x, -int((window_size - 1) / 2), axis=1)

    sobel_y = np.roll(sobel_y, -int((window_size - 1) / 2), axis=0)
    sobel_y = np.roll(sobel_y, -int((window_size - 1) / 2), axis=1)
    
    return sobel_x, sobel_y


def corner_detection(input_im, threshold=10, window_size=3, harris_k_value=0.03):
    # Math for this (Harris corner detection) taken directly from "A COMBINED CORNER AND EDGE DETECTOR" by Chris Harris and Mike Stephens
    sobel_x, sobel_y = sobel_filter(input_im, window_size=3)

    #print(f'Sobel X: {np.min(sobel_x)}, {np.max(sobel_x)}')
    #print(f'Sobel Y: {np.min(sobel_y)}, {np.max(sobel_y)}')
    
    i_x_2 = np.power(sobel_x, 2)
    i_y_2 = np.power(sobel_y, 2)
    i_x_y = sobel_x * sobel_y

    #print(f'i_x_y min/max: {np.min(i_x_y)}/{np.max(i_x_y)}')

    # We can use a kernel to do the windowing between the operators.
    # Again, use fft's to ensure we keep the scale of the values and don't drop negative values
    window_kernel = np.ones(window_size * window_size)
    i_x_2 = convolve_with_fft(i_x_2, window_kernel, window_size)
    i_y_2 = convolve_with_fft(i_y_2, window_kernel, window_size)
    i_x_y = convolve_with_fft(i_x_y, window_kernel, window_size)

    # Math for this (Harris corner detection) taken directly from "A COMBINED CORNER AND EDGE DETECTOR" by Chris Harris and Mike Stephens
    response_values = ((i_x_2 * i_y_2) - np.power(i_x_y, 2)) - (harris_k_value * np.power((i_x_2 + i_y_2), 2))

    response_values[response_values < threshold] = 0
    # Normalize the response for everything above the threshold
    response_values[response_values >= threshold] = (response_values[response_values >= threshold] / np.max(response_values))

    output_im = Image.fromarray(np.uint8(response_values * 255), mode="L")
    # Shift response back into alignment from fft window shift:
    output_im = output_im.rotate(0, translate=(-int((window_size + 1) / 2), -int((window_size + 1) / 2)), fillcolor=0)

    return output_im


def non_max_suppression(input_image, window_size=3):
    return_image = np.ones(input_image.shape)
    window_offset = int((window_size - 1) / 2)
    #for h in range(window_offset, input_image.shape[0] - window_offset):
    #    for w in range(window_offset, input_image.shape[1] - window_offset):
    #        window_compared = input_image[h - window_offset:h + window_offset, w - window_offset:w + window_offset]
    #        if ((np.sum(window_compared) > 0) and (np.max(window_compared) == input_image[h, w])):
    #            return_image[h, w] = 0

    # Made a vectorized version as I was dissatisfied with the above implementation's speed
    # Make an array with window_size x window_size channels where each represents a different offset of pixels from the original
    # roll the numpy array across the width and height relative to the window position.
    # Compare all 25 layers to find what the max value is in this window
    # Mask the original image with this max and only return the max values
    max_vals = np.ones((input_image.shape[0], input_image.shape[1], window_size*window_size))
    index_counter = 0
    for h in range(-window_offset, window_offset + 1):
        for w in range(-window_offset, window_offset + 1):
            #print(f'{h+window_offset}, {w+window_offset} = {index_counter}')
            max_vals[:, :, index_counter] = np.roll(input_image, shift=h, axis=0)
            max_vals[:, :, index_counter] = np.roll(max_vals[:, :, index_counter], shift=w, axis=1)
            index_counter += 1
    
    max_vals = np.max(max_vals, axis=2)
    return_image[(input_image == max_vals) & (input_image > 0)] = 0

    return return_image


def hough_line_detection(edges_blank_im, theta_range, rho_range, num_lines=90):
    # Initialize some baseline values we want to persist through the function
    # This stores detected edges in the image
    edges_np = np.array(edges_blank_im)
    # This will store the maximal values in Hough space we want to find
    max_points = np.zeros((num_lines, 2))
    # This is so we can vectorize the theta calculations
    theta_arange = np.arange(theta_range[0], theta_range[1], theta_range[2])
    # Unused in this version, hold-over from when we didn't used fixed rho boxes on a per-pixel basis
    rho_offset = int(((rho_range[2] - 1) / 2))

    # Extract the number of lines requested:
    for i in tqdm(range(0, num_lines)):
        # This stores the voting for each line in the hough space
        hough_space = np.zeros((int(theta_range[1] / theta_range[2]), int(rho_range[1] / rho_range[2])), dtype=int)
        # Get the edge points to compute
        edge_points = np.argwhere(edges_np == 0)
        #print(f'{theta_arange.shape} | {edge_points.shape}')
        # Vectorize the two theta values to speed up computation
        theta_values = np.tile(theta_arange, (edge_points.shape[0], 1)).T
        #print(f'{np.multiply(np.array([edge_points[:, 0], edge_points[:, 0]]), theta_values)}')

        # Compute the hough transform in a vectorized way for each edge point found
        y_array = np.tile(edge_points[:, 0], (int(theta_range[1] / theta_range[2]), 1))
        x_array = np.tile(edge_points[:, 1], (int(theta_range[1] / theta_range[2]), 1))        
        rho_vals = np.absolute(np.add(np.multiply(x_array, np.cos(np.radians(theta_values))), np.multiply(y_array, np.sin(np.radians(theta_values)))))

        #print(f'{rho_vals.shape}\n{rho_vals}')
        #print(f'{hough_space.shape}')

        # Apply the voting from each line to the hough_space matrix
        #hough_mask = np.zeros(hough_space.shape, dtype=np.bool)
        #print(f'{hough_mask[0, :].shape} {rho_vals[0, :].shape}')
        #np.put(hough_mask[0, :], (rho_vals[0, :] / rho_range[2]).astype(int), True)
        #np.put(hough_mask[1, :], (rho_vals[1, :] / rho_range[2]).astype(int), True)
        np.add.at(hough_space[0, :], (rho_vals[0, :] / rho_range[2]).astype(int), 1)
        np.add.at(hough_space[1, :], (rho_vals[1, :] / rho_range[2]).astype(int), 1)

        # Extract out the maximal point
        max_points[i, :] = np.unravel_index(np.argmax(hough_space), hough_space.shape)
        #print(f'{max_points[i, :]} = {np.max(hough_space)}')
        #print(f'Before: {hough_space[int(max_points[i, 0]), int(max_points[i, 1])]}')
        #hough_space[int(max_points[i, 0]), int(max_points[i, 1])] = 0
        #print(f'After: {hough_space[int(max_points[i, 0]), int(max_points[i, 1])]}')
        
        # Remove the maximal edges that were used and recreate the hough transform
        x = (np.round(max_points[i, 1] * rho_range[2] * np.cos(np.radians(max_points[i, 0] * theta_range[2])))).astype(np.int)
        y = (np.round(max_points[i, 1] * rho_range[2] * np.sin(np.radians(max_points[i, 0] * theta_range[2])))).astype(np.int)

        if (int(x) == 0):
            edges_np[y, :] = 1
        elif (int(y) == 0):
            edges_np[:, x] = 1

    #max_vals = np.where(hough_space >= threshold)
    #return max_vals

    return max_points, hough_space


if __name__ == '__main__':
    if(len(sys.argv) < 3):
        raise Exception(f"ERROR: provide arguments in the format like: \"python3 grade.py form.jpg output.txt\"")
    
    # This controls the extra plots that are output in report_plots, not the scored.png output
    plotting_enabled = False

    blank_loc = "test-images/blank_form.jpg"
    output_file = str(sys.argv[2])
    output_image_name = "scored.jpg"
    #output_image_name = "scored.png"

    form_im = Image.open(sys.argv[1])
    form_im = ImageOps.grayscale(form_im)
    form_orig_np = np.array(ImageOps.invert(form_im))
    form_orig_im = form_im.convert(mode="RGB")


    #blank_im = Image.open(blank_loc)
    #blank_im = ImageOps.grayscale(blank_im)

    # blank out the first 500 pixels to avoid detections in the text at the top of the page
    #draw_func = ImageDraw.Draw(blank_im)
    #draw_func.rectangle(((0, 0), (blank_im.size[0], 500)), fill=255)
    form_draw_func = ImageDraw.Draw(form_im)
    form_draw_func.rectangle(((0, 0), (form_im.size[0], 500)), fill=255)

    # 90 ensures we are only checking vertical and horizontal lines
    theta_discretization = 90
    rho_discretization = 1
    num_lines = 120

    #print(f'Performing sobel operations on blank form image')
    #sobel_x, sobel_y = sobel_filter(blank_im, 3)
    #edges_blank_im = Image.fromarray(np.uint8(np.absolute(sobel_x) + np.absolute(sobel_y)), mode='L')

    print(f'Performing sobel operations on passed in form image')
    sobel_x, sobel_y = sobel_filter(form_im, 3)
    edges_form_im = Image.fromarray(np.uint8(np.absolute(sobel_x) + np.absolute(sobel_y)), mode='L')

    #blank_im = Image.fromarray(edges_blank_np*255, mode="L")
    #form_im = Image.fromarray(edges_form_np*255, mode="L")

    #ImageDraw.floodfill(blank_im, (1, 1), 0, thresh=50)
    #ImageDraw.floodfill(form_im, (1, 1), 0, thresh=50)

    #print(f'Performing Hough transform to find lines in blank form')
    #edges_blank_np = np.array(edges_blank_im)
    #edges_blank_np[edges_blank_np <= 55] = 1
    #edges_blank_np[edges_blank_np > 55] = 0
    #blank_diagonal_dist = (np.ceil(np.sqrt(edges_blank_np.shape[0]**2 + edges_blank_np.shape[1]**2))).astype(int)
    #blank_line_intersects, blank_hough_space = hough_line_detection(edges_blank_np, theta_range=(0, 180, theta_discretization), rho_range=(0, blank_diagonal_dist, rho_discretization), num_lines=num_lines)
    #print(f'Number of intersects found in blank form: {blank_line_intersects.shape}')

    print(f'Performing Hough transform to find lines in passed in form')
    edges_form_np = np.array(edges_form_im)
    edges_form_np[edges_form_np <= 55] = 1
    edges_form_np[edges_form_np > 55] = 0
    form_diagonal_dist = (np.ceil(np.sqrt(edges_form_np.shape[0]**2 + edges_form_np.shape[1]**2))).astype(int)
    form_line_intersects, form_hough_space = hough_line_detection(edges_form_np, theta_range=(0, 180, theta_discretization), rho_range=(0, form_diagonal_dist, rho_discretization), num_lines=num_lines)
    print(f'Number of intersects found in form: {form_line_intersects.shape}')

    # Debug statement for visualizing the hough space of the transform
    #blank_im = Image.fromarray((blank_hough_space / np.max(blank_hough_space)), mode="L")
    #form_im = Image.fromarray((form_hough_space / np.max(form_hough_space)), mode="L")
    #print(blank_line_intersects)
    #print(form_line_intersects)

    #blank_draw_func = ImageDraw.Draw(blank_im)
    form_draw_func = ImageDraw.Draw(form_im)

    
    #for point in blank_line_intersects:
    #    y = np.round(point[1] * rho_discretization * np.cos(np.radians(point[0] * theta_discretization)))
    #    x = np.round(point[1] * rho_discretization * np.sin(np.radians(point[0] * theta_discretization)))
    #    #print(f'({point[0] * theta_discretization}, {point[1] * rho_discretization}) = ({y}, {x})')
    #    # Debug drawing starting point, this only handles horizontal and vertical lines currently
    #    #draw_func.point((y, x), fill=0)
    #    if (int(x) == 0):
    #        blank_draw_func.line(((y, x), (y, blank_im.size[1])), fill=0)
    #    elif (int(y) == 0):
    #        blank_draw_func.line(((y, x), (blank_im.size[0], x)), fill=0)

    # Pull out all the vertical lines 
    vertical_line_locs = []
    for point in form_line_intersects:
        x = np.round(point[1] * rho_discretization * np.cos(np.radians(point[0] * theta_discretization)))
        y = np.round(point[1] * rho_discretization * np.sin(np.radians(point[0] * theta_discretization)))
        #print(f'({point[0] * theta_discretization}, {point[1] * rho_discretization}) = ({y}, {x})')
        # Debug drawing starting point, this only handles horizontal and vertical lines currently
        #draw_func.point((y, x), fill=0)
        if (int(y) == 0):
            #form_draw_func.line(((x, y), (x, form_im.size[1])), fill=0)
            vertical_line_locs.append(x)
        #elif (int(y) == 0):
        #    form_draw_func.line(((y, x), (form_im.size[0], x)), fill=0)

    # Debug statement: plot the search line
    form_draw_func.line(((0, form_im.size[1]), (form_im.size[0], form_im.size[1])), fill=0)
    # Add the min and max values to simplify the search process
    vertical_line_locs.append(0)
    vertical_line_locs.append(form_im.size[0])
    vertical_line_locs = [int(i) for i in vertical_line_locs]
    vertical_line_locs.sort()
    #print(vertical_line_locs)

    # Filter down the vertical lines to remove false-positives
    blank_area_min_width = 10
    form_np = np.array(ImageOps.invert(form_im))
    for start, end in zip(vertical_line_locs[0:-1], vertical_line_locs[1:]):
        #print(f'{start} | {end}')
        #print(f'Area of slice = {(end - (start + 1)) * (form_im.size[1] + 1)}')
        area_of_slice = (end - (start + 1)) * (form_im.size[1] + 1)
        slice_width = (end - (start + 1))
        if ((area_of_slice == 0) or (blank_area_min_width >= slice_width)):
            #print(f'{start}, {end} skipping lines')
            continue
        #print(f'Area of slice if all white = {(end - (start + 1)) * (form_im.size[1] + 1) * 255}')
        #print(f'Sum of slice from {start} to {end} = {np.sum(form_np[start+1:end, 0:form_im.size[1]+1]) / 255}')
        percent_filled = ((np.sum(form_np[0:form_im.size[1]+1, start+1:end]) / 255) / ((end - (start + 1)) * (form_im.size[1] + 1))) * 100
        #print(f'{start}, {end} percent: {percent_filled}%')
        
        if (percent_filled < 6):
            form_draw_func.rectangle(((start, 0), (end, form_im.size[1])), outline=128, fill=128)

    
    # Parse out the columns found from the new image
    column_locs = []
    #form_draw_func.rectangle(((0, 0), (form_im.size[0], 500)), outline=0, fill=0)
    #form_im = Image.fromarray(np.array(form_im)[0:1, 0:form_im.size[1]], mode="L")
    blank_areas_found = np.where(np.array(form_im)[1, 0:form_im.size[1]] == 255)
    gray_areas_found = np.where(np.array(form_im)[1, 0:form_im.size[1]] == 128)
    blank_areas_found = np.array(blank_areas_found)
    gray_areas_found = np.array(gray_areas_found)
    column_locs_diff_blank = blank_areas_found[:, 1:] - blank_areas_found[:, :-1] - 1
    column_locs_diff_gray = gray_areas_found[:, 1:] - gray_areas_found[:, :-1] - 1
    
    column_locs.append(blank_areas_found[0, -1])
    [column_locs.append(value) for value in blank_areas_found[np.where(column_locs_diff_blank != 0)]]
    [column_locs.append(value + 1) for value in gray_areas_found[np.where(column_locs_diff_gray != 0)]]
    column_locs.sort()
    #print(f'Total columns detected: {len(column_locs) / 2}')

    # These are per-letter threshold values
    #box_edges_thresholds = [8000, 8000, 8000, 8000, 8000]
    # Threshold to determine where rows are
    box_edges_threshold = 8
    # Threshold to determine when a box is considered marked
    marked_box_threshold = 80
    # Threshold to determine when a character is written to the side
    letter_threshold = 20000
    # Offset to account for the vertical line at the top of some pages
    blank_space_offset = 5

    # Find the row locations to search for marked boxes
    hist_all_np = np.mean(form_np[:, :], axis=1)
    # Compute the cumulative sum across the height of the column
    hist_all_np = np.cumsum(hist_all_np)
    # Set a variable sized window to perform a sliding window average (and then normalize those results based on the window size)
    window_size = 5
    hist_all_np = hist_all_np[window_size:] - hist_all_np[:-window_size]
    hist_all_np = hist_all_np[window_size - 1:] / window_size
    row_locs_all = np.array(np.where(hist_all_np > box_edges_threshold)[0])
    added_zero_values = [row_locs_all[0], row_locs_all[-1]]
    threshold_vals_all = np.array(row_locs_all[:-1] - row_locs_all[1:] + 1)
    row_locs_all = row_locs_all[np.where(threshold_vals_all != 0)[0]].tolist()
    row_locs_all.extend(added_zero_values)
    row_locs_all = np.flip(np.sort(np.array(row_locs_all), kind='mergesort'))
    #print(f'Number of column separations: {len(row_locs_all)}')

    # Only look at the first 29 rows, truncate everything else as they aren't valid entries
    row_locs_all = row_locs_all[:30]

    if plotting_enabled:
        plt.figure()
        plt.plot(np.arange(0, len(hist_all_np)), hist_all_np, color='cornflowerblue', label='Pixel Row Response')
        plt.plot(np.arange(0, len(hist_all_np)), np.ones(len(hist_all_np)) * box_edges_threshold, color='red', label='Box Edge Threshold')
        plt.scatter(row_locs_all, np.ones(len(row_locs_all)), color='orange', marker='|', label='Min peaks')
        plt.gcf().set_dpi(300)
        plt.legend(loc='upper left')
        plt.savefig(f'report_plots/hist_all_columns.png')

    form_draw_func = ImageDraw.Draw(form_orig_im)

    # Only the first 29/29/27 rows are used for each answer group.
    answer_array = np.zeros((15, 29))
    column_counter = 0
    letters_beside = np.zeros((3, 29))
    # This is the left shift offset from the start of the first column to look for letters
    offset_box_width = 50

    # Determine values for each column searching from the bottom of the page to the top
    for column_index, (c_start, c_end) in tqdm(enumerate(zip(range(0, len(column_locs), 2), range(1, len(column_locs), 2))), total=(len(column_locs) / 2)):
        col_start = column_locs[c_start]
        col_end = column_locs[c_end] + 1

        #print(f'{column_locs[start]}, {column_locs[end]}')
        # average across the width of each column
        hist_np = np.mean(form_np[:, col_start:col_end], axis=1)
        # Compute the cumulative sum across the height of the column
        hist_np = np.cumsum(hist_np)
        # Set a variable sized window to perform a sliding window average (and then normalize those results based on the window size)
        window_size = 10
        hist_np = hist_np[window_size:] - hist_np[:-window_size]
        hist_np = hist_np[window_size - 1:] / window_size

        row_median_vals = []
        for row_index, (r_end, r_start) in enumerate(zip(range(0, len(row_locs_all), 1), range(1, len(row_locs_all), 1))):
            row_start = row_locs_all[r_start] + blank_space_offset
            row_end = row_locs_all[r_end] + 1 + blank_space_offset
            row_median_val = np.median(hist_np[row_start:row_end])
            if (row_median_val >= marked_box_threshold):
                row_median_vals.append(((row_start + row_end) / 2, row_median_val, row_start, row_end))
                answer_array[column_index, row_index] = 1
            
            if ((column_index % 5) == 0):
                total_filled_pixels_in_letter_space = np.sum(form_orig_np[row_start:row_end, col_start-offset_box_width-60:col_start-offset_box_width])

                if (total_filled_pixels_in_letter_space > letter_threshold):
                    form_draw_func.rectangle(((col_start-offset_box_width-60, row_start), (col_start-offset_box_width, row_end)), outline=(255, 128, 0), width=3)
                    letters_beside[column_counter, row_index] = 1
                
                #print(f'Searching row {row_index} in column {column_counter} | Total pixels: {total_filled_pixels_in_letter_space}')
        
        if ((column_index % 5) == 0):
            column_counter += 1
        
        row_median_vals.reverse()
        row_median_vals = np.asarray(row_median_vals)

        # Code used to make histogram plots of each column
        if plotting_enabled:
            plt.figure()
            plt.plot(np.arange(0, len(hist_np)), hist_np, color='cornflowerblue', label='Pixel Row Response')
            plt.plot(np.arange(0, len(hist_np)), np.ones(len(hist_np)) * marked_box_threshold, color='limegreen', label='Marked Box Threshold')
            plt.scatter(row_locs_all, np.ones(len(row_locs_all)), color='orange', marker='|', label='Min peaks')
            if (row_median_vals.size != 0):
                plt.scatter(row_median_vals[:, 0], np.ones(len(row_median_vals[:, 0])) * -1, color='limegreen', marker='+', label='Marked Points')

            plt.gcf().set_dpi(300)
            plt.legend(loc='upper left')
            plt.savefig(f'report_plots/hist_{column_index}.png')

        # Had to break this out as we always want to display the scored.png image, but don't always want the debugging plots
        if (row_median_vals.size != 0):
            for row in row_median_vals:
                form_draw_func.rectangle(((col_start, row[2]), (col_end, row[3])), outline=(0, 225, 15), width=3)

    # Flip this array back to top-down instead of bottom-up
    letters_beside = np.flip(letters_beside, axis=1)

    
    #print(f'First column answers (1-29): {np.flip(answer_array[0:5, :29], axis=1).T}')
    #print(f'Second column answers (30-58): {np.flip(answer_array[5:10, :29], axis=1).T}')
    #print(f'Third column answers (59-85): {np.flip(answer_array[10:15, :29], axis=1)[:, :27].T}')

    # Output to text file
    answer_counter = 1
    with open(output_file, "w") as output_fd:
        for row, letter_present in zip(np.flip(answer_array[0:5, :29], axis=1).T, letters_beside[0, :29]):
            output_string = f'{answer_counter} '
            if row[0] == 1:
                output_string += 'A'
            if row[1] == 1:
                output_string += 'B'
            if row[2] == 1:
                output_string += 'C'
            if row[3] == 1:
                output_string += 'D'
            if row[4] == 1:
                output_string += 'E'
            if letter_present == 1:
                output_string += ' x'
            output_string += '\n'
            output_fd.write(output_string)
            #print(output_string[:-1])
            answer_counter += 1

        for row, letter_present in zip(np.flip(answer_array[5:10, :29], axis=1).T, letters_beside[1, :29]):
            output_string = f'{answer_counter} '
            if row[0] == 1:
                output_string += 'A'
            if row[1] == 1:
                output_string += 'B'
            if row[2] == 1:
                output_string += 'C'
            if row[3] == 1:
                output_string += 'D'
            if row[4] == 1:
                output_string += 'E'
            if letter_present == 1:
                output_string += ' x'
            output_string += '\n'
            output_fd.write(output_string)
            #print(output_string[:-1])
            answer_counter += 1

        for row, letter_present in zip(np.flip(answer_array[10:15, :29], axis=1)[:, :27].T, letters_beside[2, :27]):
            output_string = f'{answer_counter} '
            if row[0] == 1:
                output_string += 'A'
            if row[1] == 1:
                output_string += 'B'
            if row[2] == 1:
                output_string += 'C'
            if row[3] == 1:
                output_string += 'D'
            if row[4] == 1:
                output_string += 'E'
            if letter_present == 1:
                output_string += ' x'
            output_string += '\n'
            output_fd.write(output_string)
            #print(output_string[:-1])
            answer_counter += 1


    #blank_im.save("blank_" + output_image_name)
    form_orig_im.save(output_image_name)
