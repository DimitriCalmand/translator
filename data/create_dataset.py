import pandas as pd

with open('europarl-v7.fr-en.en', 'r', encoding='utf-8') as file1, open('europarl-v7.fr-en.fr', 'r', encoding='utf-8') as file2:
    lines_file1 = file1.readlines()
    lines_file2 = file2.readlines()

lines_file1 = [line.strip() for line in lines_file1]
lines_file2 = [line.strip() for line in lines_file2]

df1 = pd.DataFrame({'en': lines_file1})
df2 = pd.DataFrame({'fr': lines_file2})

result_df = pd.concat([df1, df2], axis=1)

result_df.to_csv('dataset.csv', index=False)

print("Concaténation terminée. Le résultat a été enregistré dans le fichier_concatene.csv.")

