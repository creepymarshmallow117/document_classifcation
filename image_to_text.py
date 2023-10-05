import os
import sys
import pdf2image
from os import walk
from os import listdir
from os.path import isfile, join
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from os.path import dirname
from zipfile import ZipFile

path = r"D:\pdf_to_image output\doc_classification_samples"

files_and_directories = os.listdir(path)

filenames = [file for file in files_and_directories if os.path.isfile(os.path.join(path, file))]



out_path = r"D:\Bizamica\document_classifcation\image_to_text_out"

for filename in filenames:
    file_path = os.path.join(path, filename)
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    print(filename)
    print('\n\n--------------------------\n\n')
    print(text)
    print('\n\n--------------------------\n\n')
    temp_out_path = os.path.join(out_path, filename)
    temp_out_path = temp_out_path[:-4] + '.txt'
    with open(temp_out_path, 'w') as file:
        file.write(text)
