import os

file_path = r"C:\Users\creep\BizAmica\document_classifcation\image_to_text_out"

file_list = os.listdir(file_path)

filenames = [file for file in file_list if os.path.isfile(os.path.join(file_path, file))]

for filename in filenames:
    f = os.path.join(file_path, filename)
    with open(f, 'r') as file:
        file_content = file.read()
        output_path = ''
        if 'airbus' in file_content.lower():
            output_path = r'C:\Users\creep\BizAmica\document_classifcation\airbus'
        else:
            output_path = r'C:\Users\creep\BizAmica\document_classifcation\non-airbus'
        output_path = os.path.join(output_path, filename)
        with open(output_path, 'w') as output_file:
            output_file.write(file_content)
