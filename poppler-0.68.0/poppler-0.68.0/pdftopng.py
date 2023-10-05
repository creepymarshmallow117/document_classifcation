from pdf2image import convert_from_path
import os

jpegopt = {"quality": 100, "progressive": True, "optimize": True}
directory = r"C:\Users\Aditya\Downloads\temp"
output_directory = r"D:\pdf_to_image output\doc_classification_samples"
filenames = os.listdir(directory)
k = 1

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i, filename in enumerate(filenames):
    print(filename)
    path = os.path.join(directory, filename)
    try:
        images = convert_from_path(path, fmt='png', jpegopt=jpegopt, grayscale=True, dpi=200, poppler_path=r'C:\Users\Aditya\Downloads\poppler-0.68.0\poppler-0.68.0\poppler-0.68.0\bin')
        
        # Save only the first image (page) as a PNG file
        # first_image = images[0]
        # output_path = os.path.join(output_directory, f"{i+1}_{filename}.png")
        # first_image.save(output_path)

        for j in range(len(images)):
        # Save pages as images in the pdf
            # images[j].save(r'sample png\\'+filename[i].split(".")[0]+"_"+ str(j) +'.png')
            #images[j].save(output_directory+filename.split(".")[0]  + str(k) + '_' + str(j) +'.png')
            images[j].save(output_directory+filename.split(".")[0]+ '_' + str(j) +'.png')
        k=k+1
        print(path)
    except:
        print("file no processed", filename)
        pass
    # for j in range(len(images)):
   
    #     # Save pages as images in the pdf
    #     # images[j].save(r'sample png\\'+filename[i].split(".")[0]+"_"+ str(j) +'.png')
    #     images[j].save(r'COO\output\\' + str(k) + '_' + str(j) +'.png')
    # k=k+1
    #     # print(path)
