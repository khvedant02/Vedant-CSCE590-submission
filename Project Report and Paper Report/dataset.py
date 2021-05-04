from EDA import readability_score, pos_tagging, clickbait, posVector
import pandas as pd
import pickle

df = pd.read_csv("data/Dataset.csv") #dataset is created from various sources which are mentioned in README

cScore = [float(clickbait(doc)) for doc in df['text'].values.tolist()]
cScore = [round(i/100,1) for i in cScore]
df['clickbait-score'] = cScore

df['pos_tagging'] = [pos_tagging(doc) for doc in df['text'].values.tolist()]

pVec = [posVector(pos_tagging(doc)) for doc in df['text'].values.tolist()]

df['NNP'] = [vec[0] for i in pVec]
df['DT'] = [vec[1] for i in pVec]
df['IN'] = [vec[2] for i in pVec]
df['JJ'] = [vec[3] for i in pVec]
df['NN'] = [vec[4] for i in pVec]
df['NNS'] = [vec[5] for i in pVec]

rSc = [readability_score(doc) for doc in df['text'].values.tolist()]

rSc = [round(i/100,1) for i in rSc]

df['redability_score'] = rSc

main_vec = list()
for i,j,k in zip(cScore, rSc, pVec):
    main_vec.append([i,j]+k)

df['fvec'] = main_vec #feature vector from EDA 

df.to_csv("data/Dataset.csv") #saving file with all the features
pickle.dump(open("data/Dataset.pkl","wb")) #saving in pickle format