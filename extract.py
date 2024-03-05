from PIL import Image
from PIL import ImageFilter
import sys

import numpy as np

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        raise Exception(f"ERROR: provide arguments in the format like: \"python3 ./extract.py injected.jpg output.txt\"")
    form_image = Image.open(sys.argv[1])
    output_file = str(sys.argv[2])
    
    letter_ranges = {
        "": 0,
        "A": 40,
        "B": 120,
        "C": 200,
        "D": 160,
        "E": 80
    }
    reverse_index = {
        0: "",
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E"
    }

    img = form_image
    width, height = img.size
    # Convert image to pixels
    pixels = np.array(img).T
    width = 160
    height = 160
    box_size = 20
    checkerboard_flip_value = 40
    checkerboard_flip = 0
    #defining a start pixel value to start encoding 
    start_x = 650
    end_x = 850
    start_y = 400
    end_y = 600
    k=0
    s_no=1
    c=0
    str1=""
    num_lines_found = 0
    lst=[]
    with open(output_file, 'w') as file:
        for i in range(650,850, 5):
            for j in range(400, 600, box_size):
                if (num_lines_found != 0) and ((num_lines_found % checkerboard_flip_value) == 0):
                    checkerboard_flip = not checkerboard_flip

                while(k<5):
                    if (((num_lines_found % 2) == checkerboard_flip)):
                        p = np.mean(pixels[i+k,j:j+box_size])
                    else:
                        p = np.mean(255 - pixels[i+k,j:j+box_size])

                    #print(f'{num_lines_found + 1} {p} | ({i}, {j})')
                    p = np.absolute(np.array(list(letter_ranges.values()) - p))
                    p_min = np.argmin(p)
                    #print(f'\t{num_lines_found + 1} {p} min is {p_min} = {reverse_index[p_min]}')

                    #if(int(np.sum(pixels[i+k,j:j+box_size])) != 0 and int(np.sum(pixels[i+k,j:j+box_size])) != 255):
                    str1=str1+reverse_index[p_min]
                    k=k+1
                num_lines_found += 1
                file.write(f"{s_no} {str1}\n")
                if (num_lines_found == 85):
                    c=1
                    break
                s_no=s_no+1
                str1=""
                k=0
            if(c==1):
                break
