import pandas as pd
import numpy as np

CSV_FILE_PATH = './all_smiles.csv'
data = pd.read_csv(CSV_FILE_PATH,encoding="Windows-1256")
data = data.dropna(axis=0,subset = ["SMILES"])
p_data = data
print(data.columns.values.tolist())
print(len(data))

'''
charset = set("".join(list(data.smiles))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.smiles]) + 5
print (str(charset))
print(len(charset), embed)


def vectorize(smiles):
    one_hot = np.zeros((smiles.shape[0], embed, len(charset)), dtype=np.int8)
    for i, smile in enumerate(smiles):
        # encode the startchar
        one_hot[i, 0, char_to_int["!"]] = 1
        # encode the rest of the chars
        for j, c in enumerate(smile):
            one_hot[i, j + 1, char_to_int[c]] = 1
        # Encode endchar
        one_hot[i, len(smile) + 1:, char_to_int["E"]] = 1
    # Return two, one for input and the other for output
    return one_hot[:, 0:-1, :], one_hot[:, 1:, :]

X, _ = vectorize(data.smiles)
print()
'''

#https://www.zhihu.com/topic/20058652/top-answers
smiles=[]
for i in data['SMILES']:
    smiles.append(i)
#print(smiles[0])
#'CCC(C)C(NC(=O)C(CC(=O)N[O-])Cc1ccccc1)C(=O)NC(CC(C)C)C(=O)[O-]'


def smi_preprocessing(smi_sequence):
    splited_smis=[]
    length=[]
    end="/n"
    begin="&"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#"]
    for i in range(len(smi_sequence)):
        smi=smi_sequence[i]
        splited_smi=[]
        j=0
        while j<len(smi):
            smi_words=[]
            if smi[j]=="[":
                smi_words.append(smi[j])
                j=j+1
                while smi[j]!="]":
                    smi_words.append(smi[j])
                    j=j+1
                smi_words.append(smi[j])
                words = ''.join(smi_words)
                splited_smi.append(words)
                j=j+1

            else:
                smi_words.append(smi[j])

                if j+1<len(smi[j]):
                    smi_words.append(smi[j+1])
                    words = ''.join(smi_words)
                else:
                    smi_words.insert(0,smi[j-1])
                    words = ''.join(smi_words)

                if words not in element_table:
                    splited_smi.append(smi[j])
                    j=j+1
                else:
                    splited_smi.append(words)
                    j=j+2

        splited_smi.append(end)
        splited_smi.insert(0,begin)
        splited_smis.append(splited_smi)
    return splited_smis

smi = smi_preprocessing(smiles)
print(smi[0])

vocalbulary=[]
for i in smi:
    vocalbulary.extend(i)
vocalbulary=list(set(vocalbulary))

print(vocalbulary, len(vocalbulary))
def smi2id(smiles,vocalbulary):
    sequence_id=[]
    for i in range(len(smiles)):
        smi_id=[]
        for j in range(len(smiles[i])):
            smi_id.append(vocalbulary.index(smiles[i][j]))
        sequence_id.append(smi_id)
    return sequence_id

docs = dict(zip(vocalbulary, range(len(vocalbulary))))
print(docs)


def one_hot_encoding(smi, vocalbulary):
    res=[]
    for i in vocalbulary:
        if i in smi:
            res.append(1)
        else:
            res.append(0)
    return res

print(one_hot_encoding(smi[0], vocalbulary))

def encode_smiles(smi,vocalbulary):
    res = []
    for drug_smile in smi:
        res.append(one_hot_encoding(drug_smile,vocalbulary))
    print(len(res))
    df = pd.DataFrame(res)
    print(df.shape)
    df['dbid'] = p_data['DrugBank ID'].values.tolist()
    df['keggid'] = p_data['KEGG Drug ID'].values.tolist()
    new_df = df.dropna(axis=0, subset=['keggid'])
    new_df.to_csv('./encoded_smiles_all.csv', encoding='utf-8',index=False)


from sklearn.decomposition import PCA
def calculate_pca(similarity_profile_file,output_file,p_data):
    pca = PCA(copy=True, iterated_power='auto', n_components=96, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)
    df = pd.read_csv(similarity_profile_file, index_col=0)

    X = df.values
    X = pca.fit_transform(X)

    new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(96)], index=df.index)
    print(new_df.shape)
    #new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(2)], index=df.index)
    new_df['dbid'] = p_data['DrugBank ID'].values.tolist()
    new_df['keggid'] = p_data['KEGG Drug ID'].values.tolist()
    print(new_df.head())
    new_df = new_df.dropna(axis=0, subset=['keggid'])
    new_df.to_csv(output_file, encoding='utf-8',index=False)
    return new_df


encode_smiles(smi,vocalbulary)
input_file = "./encoded_smiles2.csv"
output_file = "./pca_smiles_kegg_96.csv"
new_data = calculate_pca(input_file, output_file,p_data)

