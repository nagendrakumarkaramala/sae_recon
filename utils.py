import pandas as pd
import numpy as np
import sys
from transformers import AutoTokenizer, AutoModel
import torch
import json
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
# import utils
# from utils import get_swap_dict, tokenizer,model,modenc, normalize, reconcile
import dateutil.parser as parser
from transformers import AutoTokenizer, AutoModel
import torch
import os
from datetime import datetime
import ast
import io
from io import BytesIO

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
df=pd.DataFrame()
df2=pd.DataFrame()
final2=pd.DataFrame()
study_src1_cols=''
study_src2_cols=''
lists=['site','subject','verbatim','decode','start_date','start_time',
       'end_date','end_time']
dtypes={"site":"string","subject":"string","verbatim":"string","decode":"string","start_date":"date","start_time":"time","end_date":"date","end_time":"time"}
merge_cols="subject,verbatim,decode,start_date,end_date"

lists2=lists



def dfHasData(df):
    return df.empty

def dfHasColumns(df):
    return len(df.columns)

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def modenc(sentences): 
    # Tokenize sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512,return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        output = model(**inputs)
    
    # Perform pooling. In this case, mean pooling.
    embeddings = meanpooling(output, inputs['attention_mask'])
    return embeddings

def normalize(df,colsdtypes):
    for key, value in colsdtypes.items():
        if value=='date':
            df[key]=df[key].apply(pd.to_datetime)
    return df    
    
 
def reconcile(st,ds1,ds2,study_src1_cols,study_src2_cols,dtypes,ssn1,ssn2):        
    study_src1_scols=study_src1_cols
    study_src2_scols=study_src2_cols
    
    # study_src1_dtypes=config.get('source1','dtypes')
    study_src1_sdtypes=dtypes
    study_src2_sdtypes=dtypes
    
    # study_src2_dtypes=config.get('source2','dtypes')
    # study_src2_sdtypes=json.loads(study_src2_dtypes)
    
    # ds1=pd.read_excel(study_src1_file,sheet_name=study_src1_sheet)
    # ds2=pd.read_excel(study_src2_file,sheet_name=study_src2_sheet)
    
    df1 = ds1.rename(columns=get_swap_dict(study_src1_scols))
    df1=df1[list(study_src1_scols.keys())]
    
    df2 = ds2.rename(columns=get_swap_dict(study_src2_scols))
    df2=df2[list(study_src2_scols.keys())]
    
    df1=normalize(df1,study_src1_sdtypes)
    df2=normalize(df2,study_src2_sdtypes)
    
    df1=df1.add_prefix(ssn1)
    df2=df2.add_prefix(ssn2)
    
    
    src1_merge_cols=[ssn1 + x for x in list(merge_cols.split(",")) if not str(x) == "nan"]
    src2_merge_cols=[ssn2 + x for x in list(merge_cols.split(",")) if not str(x) == "nan"]
    
    #-------------------------------------------------------------------------------------------------
    # Start : Matching records
    #-------------------------------------------------------------------------------------------------
    finalx=pd.merge(df2,
                    df1,
                    left_on=src2_merge_cols,
                    right_on=src1_merge_cols,
                    how='outer',indicator=True)
    matched=finalx[finalx['_merge']=='both']
    matched=matched.drop_duplicates()
    matched=matched.reset_index()
    matched['Message']='Reconciled'
    #-------------------------------------------------------------------------------------------------
    # End : Matching records
    #-------------------------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------------------------
    # Start : Missing clinical data
    #-------------------------------------------------------------------------------------------------
    missclin=pd.merge(df2,
                    df1,
                    left_on=[ssn2+'subject'],
                    right_on=[ssn1+'subject'],
                    how='left',indicator=True)
    missclin=missclin[missclin['_merge']=='left_only']
    missclin['Message']='Missing Clinical Record.'
    missclin=missclin.replace(np.NaN,'')
    missclin=missclin.drop_duplicates()
    missclin=missclin.reset_index()
    #-------------------------------------------------------------------------------------------------
    # End : Missing clinical data
    #-------------------------------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------------------------------
    # Start : Missing safety data
    #-------------------------------------------------------------------------------------------------
    misssaety=pd.merge(df1,
                    df2,
                    right_on=[ssn2+'subject'],
                    left_on=[ssn1+'subject'],
                    how='left',indicator=True)
    misssaety=misssaety[misssaety['_merge']=='left_only']
    misssaety=misssaety.replace(np.NaN,'')
    misssaety=misssaety.drop_duplicates()
    misssaety=misssaety.reset_index()
    misssaety['Message']='Missing Safety Record.'
    #-------------------------------------------------------------------------------------------------
    # End : Missing safety data
    #-------------------------------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------------------------------
    # Start : Dates mismatch
    #-------------------------------------------------------------------------------------------------
    
    pv2=pd.merge(df2,
                  matched,
                  on=src2_merge_cols,
                  how='left')
    pv2=pv2[pv2['_merge']=='left_only']
    pv2=pv2.drop_duplicates()
    pv2=pv2.reset_index()
    
    pv_dates_mismatched=pd.merge(df2,
                          df1,
                          left_on=[ssn2+'subject',ssn2+'decode'],
                          right_on=[ssn1+'subject',ssn1+'decode'],
                          how='left',indicator=True)
    
    pv_dates_mismatched=pv_dates_mismatched[
                        ((pv_dates_mismatched[ssn2+'start_date']==pv_dates_mismatched[ssn1+'start_date']) & (pv_dates_mismatched[ssn2+'end_date']!=pv_dates_mismatched[ssn1+'end_date']))
                        |
                        ((pv_dates_mismatched[ssn2+'start_date']!=pv_dates_mismatched[ssn1+'start_date']) & (pv_dates_mismatched[ssn2+'end_date']==pv_dates_mismatched[ssn1+'end_date']))
                        ]
    
    pv_dates_mismatched['Message']=pv_dates_mismatched.apply(lambda x: 'Start Dates mismatch.' 
                                                              if x[ssn2+'start_date']!=x[ssn1+'start_date'] 
                                                              else 'End Dates mismatch.'
                                                              if x[ssn2+'end_date']!=x[ssn1+'end_date']
                                                              else '', axis=1)
    #-------------------------------------------------------------------------------------------------
    # End: Dates mismatch
    #-------------------------------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------------------------------
    # Start : Terms mismatch.AI Based Terms matched.
    #-------------------------------------------------------------------------------------------------
    
    pv_ai_term_match=pd.merge(df2,
                                pv_dates_mismatched[[ssn2+'subject', ssn2+'decode',ssn2+'start_date',ssn2+'end_date']],
                                how='left',indicator=True)
    pv_ai_term_match=pv_ai_term_match[pv_ai_term_match['_merge']=='left_only']
    pv_ai_term_match.sort_values(ssn2+'start_date',inplace=True)
    
    df1=df1[~df1[ssn1+'start_date'].isnull()]
    df1.sort_values(ssn1+'start_date',inplace=True)
    
    pv_ai_term_match2=pd.merge_asof(pv_ai_term_match,
                                df1,
                                left_on=ssn2+'start_date',
                                right_on=ssn1+'start_date',
                                direction="nearest", left_by=ssn2+'subject', right_by=ssn1+'subject')
    
    pv_ai_term_match2=pv_ai_term_match2.replace(np.NaN,'')
    pv_ai_term_match2=pv_ai_term_match2[(pv_ai_term_match2[ssn2+'decode']!='') & (pv_ai_term_match2[ssn1+'decode']!='')]
    pv_ai_term_match2['pv_Emb']=pv_ai_term_match2.apply(lambda x: modenc([x[ssn2+'verbatim'],x[ssn2+'decode']]),axis=1)
    pv_ai_term_match2['clin_Emb']=pv_ai_term_match2.apply(lambda x: modenc([x[ssn1+'verbatim'],x[ssn1+'decode']]),axis=1)
    pv_ai_term_match2['SimScore']=pv_ai_term_match2.apply(lambda x: torch.cosine_similarity(x['pv_Emb'],x['clin_Emb'])[0],axis=1)
    pv_ai_term_match2['SimScore']=pv_ai_term_match2['SimScore'].astype(float)
    
    pv_ai_term_match2['pv_Max_SimScore']=pv_ai_term_match2.groupby([ssn2+'subject',ssn2+'decode'])['SimScore'].transform(max)
    pv_ai_term_match2=pv_ai_term_match2[pv_ai_term_match2['SimScore']==pv_ai_term_match2['pv_Max_SimScore']]
    pv_ai_term_match2['Message']='Terms mismatch.AI Based Terms matched.'
    pv_ai_term_match2['Message']=pv_ai_term_match2.apply(lambda x: 'Terms mismatch.AI Based Terms matched. End dates mismatch or missing.'
                                                          if x[ssn2+'end_date']!=x[ssn1+'end_date'] 
                                                          else x['Message'], axis=1)
    
    #-------------------------------------------------------------------------------------------------
    # End: Terms mismatch.AI Based Terms matched.
    #-------------------------------------------------------------------------------------------------
    pv_res=pd.concat([matched,missclin,misssaety,pv_dates_mismatched,pv_ai_term_match2])
    
    final=pv_res
    
    missclinterm=pd.merge(df2,
                    df1,
                    left_on=[ssn2+'subject',ssn2+'decode'],
                    right_on=[ssn1+'subject',ssn1+'decode'],
                    how='left',indicator=True)
    missclinterm=missclinterm[missclinterm['_merge']=='left_only']
    missclinterm=missclinterm.replace(np.NaN,'')
    missclinterm=missclinterm.drop_duplicates()
    missclinterm=missclinterm.reset_index()
    
    missclinterm2=pd.merge(missclinterm,
                            pv_ai_term_match2[[ssn2+'subject',ssn2+'decode']],
                            left_on=[ssn2+'subject',ssn2+'decode'],
                            right_on=[ssn2+'subject',ssn2+'decode'],
                            how='left')
    missclinterm2=missclinterm2[missclinterm2['_merge']=='left_onlyy']   
    
    missclinterm3=pd.merge(missclinterm2[[ssn2+'subject',ssn2+'verbatim', ssn2+'decode',ssn2+'start_date',ssn2+'end_date',
                                ssn1+'subject',ssn1+'verbatim',ssn1+'decode',ssn1+'start_date',ssn1+'end_date']],
                            missclin[[ssn2+'subject',ssn2+'decode']],
                            left_on=[ssn2+'subject',ssn2+'decode'],
                            right_on=[ssn2+'subject',ssn2+'decode'],
                            how='left',indicator=True)
    missclinterm3=missclinterm3[missclinterm3['_merge']=='left_only']   
    missclinterm3['Message']='Missing Clinical Term.'
    
    misssaetyterm=pd.merge(df1[[ssn1+'subject',ssn1+'verbatim',ssn1+'decode',ssn1+'start_date',ssn1+'end_date']],
                    df2[[ssn2+'subject',ssn2+'verbatim', ssn2+'decode',ssn2+'start_date',ssn2+'end_date']],
                    right_on=[ssn2+'subject', ssn2+'decode'],
                    left_on=[ssn1+'subject', ssn1+'decode'],
                    how='left',indicator=True)
    misssaetyterm=misssaetyterm[misssaetyterm['_merge']=='left_only']
    
    misssaetyterm=misssaetyterm.replace(np.NaN,'')
    misssaetyterm=misssaetyterm.drop_duplicates()
    misssaetyterm=misssaetyterm.reset_index()
    
    misssaetyterm2=pd.merge(misssaetyterm[[ssn1+'subject',ssn1+'verbatim',ssn1+'decode',ssn1+'start_date',ssn1+'end_date']],
                    pv_ai_term_match2[[ssn1+'subject',ssn1+'decode']],
                    left_on=[ssn1+'subject',ssn1+'decode'],
                    right_on=[ssn1+'subject',ssn1+'decode'],
                    how='left',indicator=True)
    misssaetyterm2=misssaetyterm2[misssaetyterm2['_merge']=='left_only']
    
    misssaetyterm3=pd.merge(misssaetyterm2[[ssn1+'subject',ssn1+'verbatim',ssn1+'decode',ssn1+'start_date',ssn1+'end_date']],
                    misssaety[[ssn1+'subject',ssn1+'decode']],
                    left_on=[ssn1+'subject',ssn1+'decode'],
                    right_on=[ssn1+'subject',ssn1+'decode'],
                    how='left',indicator=True)
    misssaetyterm3=misssaetyterm3[misssaetyterm3['_merge']=='left_only']
    misssaetyterm3['Message']='Missing Safety Term.'
    
    final2=pd.concat([final,missclinterm3,misssaetyterm3])
    
    final2=final2[[ssn2+'subject',ssn1+'subject',ssn2+'verbatim',ssn1+'verbatim',ssn2+'decode',ssn1+'decode',ssn2+'start_date',ssn1+'start_date',ssn2+'end_date',ssn1+'end_date','Message','SimScore']]
    
    final2[ssn2+'start_date'] = pd.to_datetime(final2[ssn2+'start_date']).dt.strftime('%Y-%m-%d')
    final2[ssn2+'end_date'] = pd.to_datetime(final2[ssn2+'end_date']).dt.strftime('%Y-%m-%d')
    final2[ssn1+'start_date'] = pd.to_datetime(final2[ssn1+'start_date']).dt.strftime('%Y-%m-%d')
    final2[ssn1+'end_date'] = pd.to_datetime(final2[ssn1+'end_date']).dt.strftime('%Y-%m-%d')
    final2=final2.replace(np.NaN,'')
    final2=final2.drop_duplicates()
    final2=final2.reset_index()
    final2 = final2.style.applymap(lambda x: f"background-color: {'#ACE5EE' if x=='Reconciled'  else '#FDE8D7' if x=='End Dates mismatch.' else '#FFF0AA' if x=='Missing Safety Record.' else '#FFEC94' if x=='Missing Safety Term.' else '#FFAEAE' if x=='Start Dates mismatch.' else '#FFFF66' if x=='Terms mismatch.AI Based Terms matched.' else '#FFFF66' if x=='Terms mismatch.AI Based Terms matched. End dates mismatch or missing.' else '#FFCC00'}", subset='Message')
    st.write(" Reconciliation Results : ")
    st.write(final2)
    
    return final2   
