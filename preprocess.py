import pandas as pd

data = pd.read_csv('Ubuntu-dialogue-corpus/dialogueText_301.csv')

def pair_up(df):
    df['answer'] = df['text'].shift(-1)

grouped= data.groupby(['folder', 'dialogueID'])


with open('dialogues', 'w') as writer:
    for key, item in grouped:
        current = grouped.get_group(key)
        pair_up(current)
        dialogue = current[['text', 'answer']]
        dialogue.dropna(inplace = True)
        for i, j in dialogue.values:
            writer.write(i + '|' + j + '\n')