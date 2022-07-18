import re
import json
import pandas as pd
from pprint import pprint

df = pd.read_excel('/content/drive/MyDrive/Rearch_Dimas/RE-DATASET/gold-standard-corpus.xlsx')
relations = list(df['relation'].unique())

relations.remove('Negative')
relation_dict = {'Negative': 0}
relation_dict.update(dict(zip(relations, range(1, len(relations) + 1))))

with open('/content/drive/MyDrive/Rearch_Dimas/RE-DATASET/rel_dict.json', 'w', encoding='utf-8') as h:
    h.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))

pprint(df['relation'].value_counts())

print("============================")
print('total data : %s' % len(df))
# print("\n")

df['rel'] = df['relation'].apply(lambda x: relation_dict[x])

texts = []

for per1, per2, text, label, e1start, e1end, e2start, e2end in zip(
        df['plant'].tolist(),
        df['disease'].tolist(),
        df['sentence'].tolist(),
        df['rel'].tolist(),
        df['e1start'].tolist(),
        df['e1end'].tolist(),
        df['e2start'].tolist(),
        df['e2end'].tolist()
):
    text = f"{text}\t{e1start}\t{e1end}\t{e2start}\t{e2end}"
    texts.append([text, label])

df = pd.DataFrame(texts, columns=['text', 'rel'])
df['length'] = df['text'].apply(lambda x: len(x))
df = df[df['length'] <= 360]

train_df = df.sample(frac=0.8, random_state=1024)
test_df = df.drop(train_df.index)
predict_df = test_df.sample(frac=0.4, random_state=1024)

with open('/content/drive/MyDrive/Rearch_Dimas/RE-DATASET/version 2/predict.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(predict_df['text'].tolist(), predict_df['rel'].tolist()):
        f.write(str(rel) + '\t' + text + '\n')
print ("\nsuccess to create predict.txt")

with open('/content/drive/MyDrive/Rearch_Dimas/RE-DATASET/version 2/train.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(train_df['text'].tolist(), train_df['rel'].tolist()):
        f.write(str(rel) + '\t' + text + '\n')
print ("success to create train.txt")

with open('/content/drive/MyDrive/Rearch_Dimas/RE-DATASET/version 2/test.txt', 'w', encoding='utf-8') as g:
    for text, rel in zip(test_df['text'].tolist(), test_df['rel'].tolist()):
        g.write(str(rel) + '\t' + text + '\n')
print ("success to create test.txt")
