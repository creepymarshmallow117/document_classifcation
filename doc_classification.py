import os
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

doc_directory = r'C:\Users\creep\BizAmica\document_classifcation\docs_by_type'

file_list = []

text_and_labels = []

for root, _, files in os.walk(doc_directory):
    for file in files:
        file_list.append(os.path.join(root, file))
        with open(os.path.join(root, file), 'r') as f:
            if 'non-airbus' in root:
                text_and_labels.append([file, 'non-airbus'])
            else:
                text_and_labels.append([file, 'airbus'])

random.shuffle(text_and_labels)

df = pd.DataFrame(text_and_labels, columns=['Text','Label'])

print(df)

plt.subplots(figsize=(15,5))
sns.countplot(data=df, x='Label', order=df['Label'].value_counts().index)

plt.show()