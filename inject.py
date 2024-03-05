from PIL import Image
from PIL import ImageFilter
import sys
import re

import numpy as np

if __name__ == '__main__':
    if(len(sys.argv) < 4):
        raise Exception(f"ERROR: provide arguments in the format like: \"\"")
    
    form_image = Image.open(sys.argv[1])
    answer_file = str(sys.argv[2])
    output_image_name = str(sys.argv[3])    

    letter_ranges = {
        "A": 40,
        "B": 120,
        "C": 200,
        "D": 160,
        "E": 80
    }

    # Open the image
    img = form_image
    #img=img.convert("RGB")
    # Get the size of the image
    width, height = img.size
    # Convert image to pixels
    #pixels = img.load()
    pixels = np.array(img).T
    # Modify pixels (for example, let's make the image grayscale)
    width = 160
    height = 160
    box_size = 20
    #defining a start pixel value to start encoding 
    start_x = 650
    end_x = 850
    start_y = 400
    end_y = 600
    k=0
    c=0
    file_path=answer_file
    lst=[]
    # printing the checker board!!
    for i in range(start_x, end_x, box_size):
       # print(i)
        for j in range(start_y, end_y, box_size):
            # Alternate between black and white boxes
            if (i // box_size + j // box_size) % 2 == 0:
                #  fill with black
                for x in range(i, min(i + box_size, end_x)):
                    for y in range(j, min(j + box_size, end_y)):
                        pixels[x, y] = 0
                        #print(pixels[x, y])
            else:
                # Fill the box with white
                for x in range(i, min(i + box_size, end_x)):
                    for y in range(j, min(j + box_size, end_y)):
                        pixels[x, y] = 255
                       
                        #print(pixels[x, y])

    # writing code to fetch the answers from the text file and store it in a dict
    alphabets_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Extracting numbers and alphabets using regular expressions
            matches = re.findall(r'(\d+)\s*([A-Za-z]+)', line)
            for match in matches:
                number, alphabets = match
                alphabets_dict[int(number)] = alphabets
    lst=list(alphabets_dict.values())
    # encoding the dict values:               

    #encoding Answers in the Checker Board!!
    for i in range(650,850, 5):
        for j  in range(400, 600, box_size):
            if(k<len(lst)):
                if(len(lst[k])==1):
                    #print(pixels[i,j:j+box_size])
                    if(int(np.sum(pixels[i,j:j+box_size])) == 0):
                        pixels[i,j:j+box_size]=letter_ranges[lst[k]]
                    elif(int(np.sum(pixels[i,j:j+box_size]) / (box_size)) == 255):
                        pixels[i,j:j+box_size]=255-letter_ranges[lst[k]]
                else:
                    for f in range(0, len(lst[k])):
                        if(int(np.sum(pixels[i+f,j:j+box_size])) == 0):
                            pixels[i+f,j:j+box_size]=letter_ranges[lst[k][f]]
                        elif(int(np.sum(pixels[i+f,j:j+box_size]) / (box_size))==255):
                            pixels[i+f,j:j+box_size]=255-letter_ranges[lst[k][f]]
                k=k+1
            else:
                c=c+1
        if(c!=0):
            k = 0
            c = 0
    
    img = Image.fromarray(pixels.T, mode="L")
    # Save the modified image as png as jpg is compressing the image
    img.save(output_image_name)