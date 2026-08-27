#!/usr/bin/env python
# coding: utf-8

# # Note
# This interactive notebook contains the code for Protein-Protein interaction prediction context which was adapted from the MHC-Peptide context. Full scripts are located in the github here: [https://github.com/jishnu-lab/SWING]

# # Imports and Data

# In[1]:


import pandas as pd # For data handling
import numpy as np
import os
import random
random.seed(42)
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sklearn
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, auc
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from Bio import SeqIO

# df = pd.read_csv('../Data/MutInt_Model/Mutation_perturbation_model.csv')

# df.shape


# In[2]:


pos_df = pd.read_csv('/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss/swing_train.pos', header=None, sep='\t')
pos_df.columns = 'refseq_id', 'Mutation', 'partner'
neg_df = pd.read_csv('/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss/swing_train.neg', header=None, sep='\t')
neg_df.columns = 'refseq_id', 'Mutation', 'partner'

df = pd.concat([pos_df, neg_df], ignore_index=True)
labels = [1] * len(pos_df) + [0] * len(neg_df)
df['Y2H_score'] = labels


# In[3]:


df


# In[4]:


fasta_dict = {
    record.description.strip(): str(record.seq)
    for record in SeqIO.parse("/data/ross/ppi_lossgain/interaction_loss/swing_train/swing_train_wt_and_vt.fasta", "fasta")
}
for key in list(fasta_dict.keys()):
    if ' ' in key:
        refseq_id, zero_based_variant = key.split(' ')
        one_based_variant = zero_based_variant[0]+str(int(zero_based_variant[1:-1])+1)+zero_based_variant[-1]
        seq = fasta_dict[key]
        del fasta_dict[key]
        fasta_dict[f'{refseq_id} {one_based_variant}'] = seq


# In[5]:


indices_to_drop = []
new_info = []
for i,row in df.iterrows():
    refseq_id, variant, partner = row['refseq_id'], row['Mutation'], row['partner']
    vt_id = f'{refseq_id} {variant}'

    # print(refseq_id, variant, partner)

    # seqs
    if refseq_id not in fasta_dict or vt_id not in fasta_dict or partner not in fasta_dict:
        indices_to_drop.append(i)
        new_info.append((None, None, None, None, None, None, None))
        continue

    target_seq = fasta_dict[refseq_id]
    interactor_seq = fasta_dict[partner]
    # print(refseq_id in fasta_dict, partner in fasta_dict, vt_id in fasta_dict)
    # if vt_id == 'NP_058132 S162G':
    #     print(target_seq[161])
    mutated_seq = fasta_dict[vt_id]

    # variant info
    before_aa = variant[0]
    position = int(variant[1:-1])
    after_aa = variant[-1]

    if before_aa == after_aa:
        indices_to_drop.append(i)
        new_info.append((None, None, None, None, None, None, None))
        continue
    
    assert target_seq[position-1] == before_aa and mutated_seq[position-1] == after_aa

    new_info.append((before_aa, position, after_aa, target_seq, interactor_seq, mutated_seq, 'Mutant'))


# In[6]:


new_cols = pd.DataFrame(new_info, columns=['Before_AA', 'Position', 'After_AA', 'Target_Seq', 'Interactor_Seq', 'Mutated_Seq (unless WT)', 'Type'])
df = pd.concat([df, new_cols], axis=1)


# In[7]:


len(df), len(indices_to_drop)


# In[8]:


if len(indices_to_drop) != 0:
    df = df.drop(indices_to_drop).reset_index(drop=True)
    indices_to_drop = []


# In[9]:


df['Position'] = df['Position'].astype(int)


# In[10]:


df.head(10)


# In[11]:


assert not df.duplicated().any(), "DataFrame contains duplicate rows"


# ### save dataset for use in pretraining with test data

# In[12]:


df.to_csv("/data/ross/ppi_lossgain/interaction_loss/sahni_fragoza_train.csv", index=False)


# ### continue

# In[13]:


# df.head(10)


# # Language Generation
# Packages needed:
# - pandas
# - numpy
# - random
# - gensim.models.doc2vec's Doc2Vec and TaggedDocument

# ## Delta Grantham Dict

# In[14]:


AA_scores = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'E':12.3,'Q':10.5,'G':9.0,'H':10.4,'I':5.2,
            'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9} # Grantham score, interchangable
AAs = list(AA_scores.keys())
aa_score_dict = {} 
for i in range(len(AAs)): # create all pairs of AAs
    for j in range(len(AAs)-i):
        AA_pair = AAs[i]+AAs[j+i]
        AA_pair_score = round(abs(AA_scores[AAs[i]]-AA_scores[AAs[j+i]])) # take rounded, absolute value of the difference of scores
        aa_score_dict[AA_pair] = AA_pair_score # forward
        aa_score_dict[AA_pair[::-1]] = AA_pair_score # and reverse


# ## Window Encodings

# ### README
# get_window_encodings() takes a pandas dataframe where each row represents a protein-protein interaction with an position of mutation (integers) column, mutated sequence column (string), and interactor sequence column (string). It also takes a window_k parameter (integer) which determines the number of amino acids of each side of the mutation to include in the mutation window. The scoring method is input as a dictionary with interacting amino acids (str) as keys and the absolute value of the rounded difference of the scores (Ex. 'AR':2). The padding score is also a parameter (integer) and should be based on the range of score differences. 
# 
# The function returns a list of score encodings strings that each represent a PPI. The ends of the encodings include padding from the sliding window process. These encodings will be broken into k-mers for the embedding model.

# In[15]:


def get_window_encodings(df, window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9): # Takes df (mut/int sequences and mutation position) and window_k (# AA's on each side of the mutation position)
    total_encodings = [] # Master list of encodings
    for i in tqdm(df.index): # Iterate through protein pairs
        pos = df[pos_colname].iloc[i]-1 # find mutation position for window
        mut_window = df[mutseq_colname].iloc[i][pos-window_k:pos+window_k+1] # Create sliding window
        interactor = df[intseq_colname].iloc[i] # Get interactor sequence
        PPI_encoding = '' # For each PPI
        its = 0 # Tracks sliding window position
        for j in range(len(interactor)): # For the entire length of the interactor
            window_scores = '' # Saves the scores between window-interactor at the 'its' position
            for k in range(len(mut_window)): # At each positon of the interactor ('its'), align mutant window and find the score differences
                try: # If 'its' is at the end of the interactor, the window is hanging off end (padding)
                    pair = mut_window[k]+interactor[k+its] 
                    score = aa_score_dict[pair]
                except: # If not a pair, its padding (end of interactor)
                    pair = None
                    score = padding_score # Padding score is 9
                window_scores = window_scores + str(score) # Add score to running string
            its +=1 # Slide down a position on the interactor
            PPI_encoding = PPI_encoding + str(window_scores) # Add to final string for interaction
        total_encodings.append(PPI_encoding) # Add to list for all interactions
    return total_encodings # List of encodings for each PPI


# ### pretrain on wt data points too???

# In[16]:


wt_seqs = []
for i in df.index: # for each mutant
    # change the mutant sequence back to wild type
    mut_seq = df.loc[i]['Mutated_Seq (unless WT)']
    before_aa = df.loc[i]['Before_AA']
    after_aa = df.loc[i]['After_AA']
    position = df.loc[i]['Position'] - 1 # POSITION IS ONE INDEXED
    if mut_seq[position] == after_aa: # check if after_AA is really at the position
        wt_seqs.append(mut_seq[:position]+before_aa+mut_seq[position+1:]) # make the wt sequence
    else:
        raise ValueError('Position Index (1 indexed) does not match Before_AA at index '+str(i))

# add WT nolabels to df and shuffle
train_wts = df.copy()
train_wts['Mutated_Seq (unless WT)']=wt_seqs
train_wts['Type']='WildType'
df_with_wts = pd.concat([df, train_wts]).sample(frac = 1, random_state = 1).reset_index(drop=True) # shuffle


# In[17]:


len(df_with_wts), len(df)


# In[18]:


# window_encodings = get_window_encodings(df, window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9)
window_encodings = get_window_encodings(df_with_wts, window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9)
# window_encodings[0] # last SW 2999999 (2*window_k+1)


# ## K-Mers

# ### README
# get_kmers_str() takes the encoding scores from get_window_encodings(), a k parameter that represents the k-mer size (integer) and the same padding_score (integer) from get_window_encodings(). 
# 
# 
# This function returns a list of lists of overlapping k-mers of specified size k, removing k-mers of only padding. Each list of k-mers are specific to each of the PPIs.

# In[19]:


def get_kmers_str(encoding_scores, k=7, padding_score=9):
    padding = {str(padding_score)} 
    for i in range(k): # Makes a set of padding scores that will be removed from the final k-mers
        padding.add(str(padding_score)*(i+1)) # {'9','99','999'...}
    kmers = [] # Master list of k-mers
    for ppi_score in tqdm(encoding_scores): # For each PPI encoding
        int_kmers = [] # K-mers specific to PPI
        for j in range(len(ppi_score)-k+1): # Iterate over the PPI encoding
            kmer = ppi_score[j:j+k] # Slice k-mers and sliding over
            if kmer not in padding: # If K-mer is just padding, don't add it
                int_kmers.append(kmer) # Keep non-padding k-mers  
        kmers.append(int_kmers) # Append k-mers to master list
    return kmers 


# In[20]:


kmers = get_kmers_str(window_encodings, k=7, padding_score=9)
len(kmers) # Should be number of interactions


# ## Get Corpus

# ### README
# get_corpus() takes in the k-mers created in get_kmers_str(). It outputs a Doc2Vec TaggedDocuments entities for each PPI to be used in a Doc2Vec model.

# In[21]:


def get_corpus(matrix, tokens_only=False):
    for i in range(len(matrix)): # for each PPI
        yield gensim.models.doc2vec.TaggedDocument(matrix[i],[i]) # Create a tagged document


# In[22]:


train_corpus = list(get_corpus(kmers))
# train_corpus[0]


# # Embedding Generation
# Training the Doc2Ved model can take over an hour. To train Doc2Vec, uncomment the follow block. Alternatively the embeddings/vectors (Doc2Vec output) are provided below with corresponding labels.
# 
# Packages needed:
# - gensim.models.doc2vec's Doc2Vec (v 4.2.0)
# 

# In[23]:


# Tuned Parameters from WandB
# D2V Training can take a long time, outputs can be loaded in following block
dim = 128
dm = 1
alpha = 0.08711
w = 6
epochs = 52

import sys
STRINGENT_PRETRAIN = int(sys.argv[1])
if STRINGENT_PRETRAIN:
    print('blind-test...')
else:
    print('test pretrain...')

if not STRINGENT_PRETRAIN:
    d2v_model = Doc2Vec.load("/data/ross/gnn/ppi_interaction_loss/SWING_sahni_fragoza_doc2vec.model")
    # d2v_model = Doc2Vec(vector_size=dim, min_count=1, alpha=alpha, dm=dm, window=w)
    # d2v_model.build_vocab(train_corpus)
    # d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=epochs) 
    print('d2v training done')

    # Get all vectors from D2V
    all_vecs = d2v_model.dv.vectors

    # d2v_model.save("/data/ross/gnn/ppi_interaction_loss/SWING_sahni_fragoza_doc2vec.model")
    df_with_wts['Vectors'] = all_vecs.tolist()

updated_df = df_with_wts[df_with_wts['Type'] == 'Mutant']
Counter(updated_df.Y2H_score.values) # interacting (0) and disrupted (1) counts


# # Classification
# Packages needed:
# - xgboost's XGBClassifier (v 1.6.1)

# ### Initializing XGBoost

# In[24]:


# Tuned Parameters from WandB
n_estimators = 375
max_depth = 6
learning_rate = 0.08966
# xgb_cl = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)


# In[25]:


# # Shuffle dataset to eliminate batch effect
# features = np.array(list(df.Vectors.values))
# labels = np.array(list(df.Y2H_score.values))

# indices = np.arange(features.shape[0]) 
# np.random.shuffle(indices)

# features = features[indices]
# labels = labels[indices]

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# print('y test:',Counter(y_test))


# In[26]:


# # Train
# xgb_cl.fit(X_train,y_train)


# In[27]:


# # Evaluate
# np.random.seed(42)
# y_test_permuted = np.random.binomial(n=1, p=(Counter(y_test)[1.0]/len(y_test)), size=[len(y_test)]) # random ys
# print('y test permuted:',Counter(y_test_permuted))

# test_pred = xgb_cl.predict(X_test)
# pred_proba = xgb_cl.predict_proba(X_test)[:,1]
# fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba)

# pred_proba_permuted = xgb_cl.predict_proba(X_test)[:,1]
# fpr_perm, tpr_perm, _ = metrics.roc_curve(y_test_permuted, pred_proba_permuted)

# auc_score = metrics.roc_auc_score(y_test,pred_proba)
# print('AUC Score:', auc_score)
# f1_score =  metrics.f1_score(y_test,test_pred)
# print('F1 Score:', f1_score)
# precision = metrics.precision_score(y_test,test_pred)
# print('Precision:', precision)
# recall = metrics.recall_score(y_test,test_pred)
# print('Recall:', recall)

# auc_score_perm = metrics.roc_auc_score(y_test_permuted, pred_proba_permuted)
# print('Perm Y AUC Score:', auc_score_perm)


# ### load gcv splits

# In[28]:


# with open('/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/fold_splits.pkl','rb') as f:
#     fold_splits = pickle.load(f)

# with open('/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/all_vt_ids.pkl','rb') as f:
#     vt_ids_splits = pickle.load(f)
# vt_ids_splits = [f"{vt_id.split(' ')[0]} {vt_id.split(' ')[1][0]}{int(vt_id.split(' ')[1][1:-1])+1}{vt_id.split(' ')[1][-1]}" for vt_id in vt_ids_splits]

# index_mapping = []
# for i,row in updated_df.iterrows():
#     vt_id = f"{row['refseq_id']}_{row['partner']} {row['Mutation']}"
#     assert vt_id in vt_ids_splits
#     index_mapping.append(vt_ids_splits.index(vt_id))

# assert len(index_mapping) == len(np.unique(index_mapping))

# ordered_df = pd.DataFrame(index=range(len(updated_df)), columns=updated_df.columns)

# for old_idx, new_idx in enumerate(index_mapping):
#     ordered_df.iloc[new_idx] = updated_df.iloc[old_idx]
# ordered_df = ordered_df.reset_index(drop=True)

# for (fold, train_idx, test_idx) in fold_splits:
#     ...


# In[29]:

macro_aucs, micro_aucs = [],[]
detailed_results = {
    'iterations': {},  # Will store results for each gcv_seed
}
for gcv_seed in range(30):
    with open(f'/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/sahni_fragoza_train_fold_splits_{gcv_seed}.pkl','rb') as f:
        fold_splits = pickle.load(f)
    fold_n_test = [len(test_idx) for _, _, test_idx in fold_splits]
    
    with open('/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/sahni_fragoza_train_all_vt_ids.pkl','rb') as f:
        vt_ids_splits = pickle.load(f)
    vt_ids_splits = [f"{vt_id.split(' ')[0]} {vt_id.split(' ')[1][0]}{int(vt_id.split(' ')[1][1:-1])+1}{vt_id.split(' ')[1][-1]}" for vt_id in vt_ids_splits]
    
    
    # In[30]:
    
    
    vt_ids_splits
    
    
    # In[31]:
    
    
    fold_splits
    
    
    # In[42]:
    
    
    len(updated_df), len(vt_ids_splits)
    
    
    # In[43]:
    
    
    index_mapping = []
    valid_indices = []
    
    for i, row in updated_df.iterrows():
        vt_id = f"{row['refseq_id']}-{row['partner']} {row['Mutation']}"
        
        if vt_id in vt_ids_splits:
            index_mapping.append(vt_ids_splits.index(vt_id))
            valid_indices.append(i)
        else:
            print(f"Skipping missing vt_id: {vt_id}")
    
    assert len(valid_indices) == len(vt_ids_splits), f'{len(valid_indices)}, {len(updated_df)}, {len(vt_ids_splits)}'
    updated_df = updated_df.loc[valid_indices].reset_index(drop=True)
    
    ordered_df = pd.DataFrame(index=range(len(updated_df)), columns=updated_df.columns)
    for old_idx, new_idx in enumerate(index_mapping):
        ordered_df.iloc[new_idx] = updated_df.iloc[old_idx]
    ordered_df = ordered_df.reset_index(drop=True)
    
    
    # In[37]:
    
    
    ordered_df
    
    
    # ### Group Cross Validation
    
    # In[41]:
    
    
    df_wt_index = []
    
    for i, row in ordered_df.iterrows():
        assert row['Type'] == 'Mutant'
        refseq_id = row['refseq_id']
        mutation = row['Mutation']
        partner = row['partner']
        y2h_score = row['Y2H_score']
    
        # Boolean mask to find the matching row in df_with_wts
        matches = df_with_wts[
            (df_with_wts['refseq_id'] == refseq_id) &
            (df_with_wts['Mutation'] == mutation) &
            (df_with_wts['partner'] == partner) &
            (df_with_wts['Y2H_score'] == y2h_score) &
            (df_with_wts['Type'] == 'WildType')
        ]
    
        assert len(matches) == 1, f"Expected 1 match, got {len(matches)} for row {i}"
        df_wt_index.append(matches.index[0])
    
    ordered_df['df_wt_index'] = df_wt_index
    
    
    
    # ### Group Cross Validation
    
    # In[51]:
    
    
    # train_idx
    
    
    # In[53]:
    
    
    # SCV can take ~20 min, can use the pre-saved DF in next cell!
    scv_df = pd.DataFrame(columns=['AUCs','PermYAUCs','F1s','Precisions','Recalls','TPRs','FPRs','FPRperms','TPRperms'])
    
    # number of iterations
    outer_loop = 10
    inner_loop = 10
    
    all_tts_aucs = []
    all_tts_permy_aucs = []
    all_tts_f1s = []
    all_tts_percisions =[]
    all_tts_recalls = []
    all_tts_fprs = []
    all_tts_tprs = []
    all_tts_permy_fprs = []
    all_tts_permy_tprs = []
    
    #  make unshuffled dataset
    # Shuffle dataset to eliminate batch effect
    features = np.array(list(ordered_df.Vectors.values))
    labels = np.array(list(ordered_df.Y2H_score.values))
    df_wt_mapping = np.array(list(ordered_df.df_wt_index.values))
    
    all_preds, all_labels_perm, all_labels = [],[],[]
    
    for (fold, train_idx, test_idx) in fold_splits:
    
    # for o_its in tqdm(range(outer_loop)): # shuffle dataset 
        # shuffle data and change the seed each time
        # np.random.seed(o_its)
        
        # create new shuffled indicies
        # indices = np.arange(features.shape[0]) 
        # indices = np.array(train_idx)
        # np.random.shuffle(indices)
        
        # features = features[indices]
        # labels = labels[indices]
    
        if STRINGENT_PRETRAIN:
            '''only pretrain on training points'''
            train_df = ordered_df.iloc[train_idx].reset_index(drop=True)
            train_df['original_index'] = train_df.index
            
            wt_seqs = []
            for i in train_df.index: # for each mutant
                # change the mutant sequence back to wild type
                mut_seq = train_df.loc[i]['Mutated_Seq (unless WT)']
                before_aa = train_df.loc[i]['Before_AA']
                after_aa = train_df.loc[i]['After_AA']
                position = train_df.loc[i]['Position'] - 1 # POSITION IS ONE INDEXED
                if mut_seq[position] == after_aa: # check if after_AA is really at the position
                    wt_seqs.append(mut_seq[:position]+before_aa+mut_seq[position+1:]) # make the wt sequence
                else:
                    raise ValueError('Position Index (1 indexed) does not match Before_AA at index '+str(i))
            
            # add WT nolabels to df and shuffle
            train_wts = train_df.copy()
            train_wts['Mutated_Seq (unless WT)']=wt_seqs
            train_wts['Type']='WildType'
            # train_df_with_wts = pd.concat([train_df, train_wts]).sample(frac = 1, random_state = 1).reset_index(drop=True) # shuffle
            train_df_with_wts = pd.concat([train_df, train_wts]).sample(frac=1, random_state=1).reset_index(drop=True)
            train_df_with_wts['doc2vec_index'] = range(len(train_df_with_wts))  # Track doc2vec order
            
            # window_encodings = get_window_encodings(ordered_df.iloc[train_idx].reset_index(drop=True), window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9)
            window_encodings = get_window_encodings(train_df_with_wts, window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9)
            kmers = get_kmers_str(window_encodings, k=7, padding_score=9)
            train_corpus = list(get_corpus(kmers))
    
            window_encodings = get_window_encodings(ordered_df.iloc[test_idx].reset_index(drop=True), window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9)
            kmers = get_kmers_str(window_encodings, k=7, padding_score=9)
            test_corpus = list(get_corpus(kmers))
            
            print(fold, 'fold', 'stringent', STRINGENT_PRETRAIN, 'pretraining doc2vec...',end='\t')
            
            d2v_model = Doc2Vec(vector_size=dim, min_count=1, alpha=alpha, dm=dm, window=w)
            d2v_model.build_vocab(train_corpus)
            d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=epochs) 
            
            # train_vectors = np.array(d2v_model.dv.vectors.tolist())
            train_df_with_wts['Vectors'] = d2v_model.dv.vectors.tolist()
    
            # Extract mutant vectors in ORIGINAL order
            mutant_df = train_df_with_wts[train_df_with_wts['Type'] == 'Mutant'].sort_values('original_index')
            train_vectors = np.array(mutant_df['Vectors'].tolist())
            
            # Assert order is correct
            assert len(train_vectors) == len(train_idx)
            assert all(mutant_df['original_index'].values == range(len(train_idx)))
    
            test_vectors = np.array([d2v_model.infer_vector(doc.words) for doc in test_corpus])
        
            X_train, X_test, y_train, y_test = train_vectors, test_vectors, labels[train_idx], labels[test_idx]
            assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
        else:
            X_train, X_test, y_train, y_test = features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]
    
        '''
        train (but not test) on wt data points
        '''
        # Collect wild-type vectors and labels.
        # STRINGENT: use fold-specific Doc2Vec WT vectors (same embedding space as train_vectors).
        # Non-STRINGENT: use global Doc2Vec WT vectors from df_with_wts.
        if STRINGENT_PRETRAIN:
            wt_df = train_df_with_wts[train_df_with_wts['Type'] == 'WildType'].sort_values('original_index')
            X_train_wt = np.array(wt_df['Vectors'].tolist())
        else:
            X_train_wt = np.array([df_with_wts.loc[idx, 'Vectors'] for idx in df_wt_mapping[train_idx]])
        y_train_wt = np.zeros(len(X_train_wt))
        
        # Concatenate with existing training data
        X_train = np.concatenate([X_train, X_train_wt])
        y_train = np.concatenate([y_train, y_train_wt])
    
        print(len(X_train), 'training points including wts',end='\t')
        
    
        # save the scores from AUCs
        tts_aucs = []   
        tts_permy_aucs = []
        tts_f1s = []
        tts_percisions =[]
        tts_recalls = []
        tts_fprs = []
        tts_tprs = []
        tts_permy_fprs = []
        tts_permy_tprs = []
        
        # for i_its in range(inner_loop): # shuffle each tts
            # TTS + perm y set
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=i_its) # change seed
        y_test_perm = np.random.binomial(n=1, p=(Counter(y_test)[1.0]/len(y_test)), size=[len(y_test)]) # random ys
    
        print(fold, 'fold', 'training XGBoost...',end='\t')
    
        # train XGB
        xgb_cl = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        xgb_cl.fit(X_train,y_train)
        # if not STRINGENT_PRETRAIN:
        #     xgb_cl.save_model(f'/data/ross/gnn/ppi_interaction_loss/SWING_xgb_fold_{fold}.json')
    
        # test XGB + permy
        test_pred = xgb_cl.predict(X_test)
        pred_proba = xgb_cl.predict_proba(X_test)[:,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba)
        tts_fprs.append(fpr)
        tts_tprs.append(tpr)
    
        test_pred_perm = xgb_cl.predict(X_test)
        pred_proba_perm = xgb_cl.predict_proba(X_test)[:,1]
        fpr_perm, tpr_perm, _ = metrics.roc_curve(y_test_perm, pred_proba_perm)
        tts_permy_fprs.append(fpr_perm)
        tts_permy_tprs.append(tpr_perm)
    
        auc_score = metrics.roc_auc_score(y_test,pred_proba)
        auc_score_perm = metrics.roc_auc_score(y_test_perm,pred_proba_perm)
        f1_score =  metrics.f1_score(y_test,test_pred)
        precision = metrics.precision_score(y_test,test_pred)
        recall = metrics.recall_score(y_test,test_pred)
        
        tts_aucs.append(auc_score)
        tts_permy_aucs.append(auc_score_perm)
        tts_f1s.append(f1_score)
        tts_percisions.append(precision)
        tts_recalls.append(recall)
            
        # Define a common set of FPR values for plotting
        common_fpr = np.linspace(0, 1, 100)
        common_fpr_perm = np.linspace(0, 1, 100)
        interp_tpr = []
        interp_tpr_perm = []
        
        # Interpolate TPR values for each fold
        for i in range(len(tts_fprs)):
            interp_tpr.append(np.interp(common_fpr, tts_fprs[i], tts_tprs[i]))
            interp_tpr_perm.append(np.interp(common_fpr_perm, tts_permy_fprs[i], tts_permy_tprs[i]))
        # Calculate the mean of the interpolated TPR values
        mean_tpr = np.mean(interp_tpr, axis=0)
        mean_tpr_perm = np.mean(interp_tpr_perm, axis=0)
    
        all_preds.extend(pred_proba)
        all_labels_perm.extend(y_test_perm)
        all_labels.extend(y_test)
    
        all_tts_aucs.append(np.mean(tts_aucs))
        all_tts_permy_aucs.append(np.mean(tts_permy_aucs))
        all_tts_f1s.append(np.mean(tts_f1s))
        all_tts_percisions.append(np.mean(tts_percisions))
        all_tts_recalls.append(np.mean(tts_recalls))
        all_tts_fprs.append(common_fpr)
        all_tts_tprs.append(mean_tpr)
        all_tts_permy_fprs.append(common_fpr_perm)
        all_tts_permy_tprs.append(mean_tpr_perm)
        print('Outer Fold #' + str(fold)+ ' Done')
    
    # save data
    scv_df['AUCs'] = all_tts_aucs
    scv_df['PermYAUCs'] = all_tts_permy_aucs
    scv_df['F1s'] = all_tts_f1s
    scv_df['Precisions'] = all_tts_percisions
    scv_df['Recalls'] = all_tts_recalls
    scv_df['FPRs'] = all_tts_fprs
    scv_df['TPRs'] = all_tts_tprs
    scv_df['FPRperms'] = all_tts_permy_fprs
    scv_df['TPRperms'] = all_tts_permy_tprs
    scv_df.to_pickle('./ross_MutInt_group_cross_validation_example_results.pkl')
    
    
    # In[54]:
    
    
    all_labels[:5], all_labels_perm[:5]
    
    
    # In[55]:
    
    
    len(all_preds), len(all_labels_perm), len(all_labels)
    
    
    # In[56]:
    
    
    # scv_df
    
    
    # In[57]:
    
    
    pair_test_classes = np.load(f'/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/swing_train_pair_test_classes_{gcv_seed}.npy')
    
    
    # In[58]:
    
    
    # Load pickle from example OR change to your SCV results
    # scv_df = pd.read_pickle('./ross_MutInt_group_cross_validation_example_results.pkl')
    
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    all_preds, all_labels_perm, all_labels = np.array(all_preds), np.array(all_labels_perm), np.array(all_labels) 
    
    
    micro_auc = []
    for pair_test_class in (1,2,3):
        print(f'c{pair_test_class}:',len(all_labels[pair_test_classes == pair_test_class]),'preds')
        roc_auc = roc_auc_score(all_labels[pair_test_classes == pair_test_class], all_preds[pair_test_classes == pair_test_class])
        micro_auc.append(roc_auc)
    
    print(f"iter {gcv_seed} micro AUC: {micro_auc}")
    # micro_aucs.append(np.array(micro_auc))
    
    # Initialize storage for this iteration
    iteration_results = {
        'folds': {}
    }
    
    curr_idx = 0
    class_auc_avgs = [0, 0, 0]
    class_counts = [0, 0, 0]
    
    for fold, n_test in enumerate(fold_n_test):
        preds = all_preds[curr_idx:curr_idx+n_test]
        labels = all_labels[curr_idx:curr_idx+n_test]
        pair_test = pair_test_classes[curr_idx:curr_idx+n_test]
        
        # Initialize storage for this fold
        fold_results = {
            'class_1': {'preds': [], 'labels': [], 'auc': None},
            'class_2': {'preds': [], 'labels': [], 'auc': None},
            'class_3': {'preds': [], 'labels': [], 'auc': None}
        }
        
        print(f'\nfold {fold}')
        for pair_test_class in (1, 2, 3):
            class_preds = preds[pair_test == pair_test_class]
            class_labels = labels[pair_test == pair_test_class]
            
            # Store predictions and labels regardless of whether AUC can be calculated
            fold_results[f'class_{pair_test_class}']['preds'] = class_preds.copy()
            fold_results[f'class_{pair_test_class}']['labels'] = class_labels.copy()
            
            # Check if we have both positive and negative samples
            n_pos = np.sum(class_labels == 1)
            n_neg = np.sum(class_labels == 0)
            
            if n_pos > 0 and n_neg > 0:
                # Valid AUC calculation
                roc_auc = roc_auc_score(class_labels, class_preds)
                print(f"c{pair_test_class} (n={len(class_preds)}): AUC-ROC: {roc_auc:.4f}", flush=True)
                
                # Store AUC
                fold_results[f'class_{pair_test_class}']['auc'] = roc_auc
                
                # Add to weighted sum
                class_auc_avgs[pair_test_class-1] += len(class_preds) * roc_auc
                class_counts[pair_test_class-1] += len(class_preds)
            else:
                # Skip this class for this fold
                print(f"c{pair_test_class} (n={len(class_preds)}): SKIPPED - insufficient pos/neg samples (pos={n_pos}, neg={n_neg})", flush=True)
                fold_results[f'class_{pair_test_class}']['auc'] = np.nan
        
        # Store fold results
        iteration_results['folds'][fold] = fold_results
        curr_idx += n_test
    
    # Calculate macro averages using only valid counts
    class_auc_avgs = np.array(class_auc_avgs)
    class_counts = np.array(class_counts)
    
    # Avoid division by zero
    macro_auc_per_class = np.zeros(3)
    for i in range(3):
        if class_counts[i] > 0:
            macro_auc_per_class[i] = class_auc_avgs[i] / class_counts[i]
        else:
            macro_auc_per_class[i] = np.nan
    
    print(f'iter {gcv_seed} class [1,2,3] macro AUC:', macro_auc_per_class)
    
    # Store iteration summary
    iteration_results['macro_auc'] = macro_auc_per_class
    iteration_results['micro_auc'] = micro_auc  # Assuming micro_auc is calculated elsewhere
    
    # Store in main results
    detailed_results['iterations'][gcv_seed] = iteration_results
    
    macro_aucs.append(macro_auc_per_class)
    micro_aucs.append(np.array(micro_auc))

        
macro_aucs = np.array(macro_aucs)
micro_aucs = np.array(micro_aucs)
stringent_code = '_no_test_pretrain' if STRINGENT_PRETRAIN else '_test_pretrain'
if len(macro_aucs) >= 30:
    np.save(f'/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/SWING_sahni_fragoza{stringent_code}_micro_aucs.npy', micro_aucs)
    np.save(f'/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/SWING_sahni_fragoza{stringent_code}_macro_aucs.npy', macro_aucs)
    
    import pickle
    with open(f'/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/SWING_sahni_fragoza{stringent_code}_detailed_results.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)

    





