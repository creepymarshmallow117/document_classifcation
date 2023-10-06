import os
import pytesseract
from PIL import Image

path = r"C:\Users\creep\BizAmica\document_classifcation\pdf_to_image"

files_and_directories = os.listdir(path)

filenames = [file for file in files_and_directories if os.path.isfile(os.path.join(path, file))]

out_path = r"C:\Users\creep\BizAmica\document_classifcation\image_to_text_out"


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
