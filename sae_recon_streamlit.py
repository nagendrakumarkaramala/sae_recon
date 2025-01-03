import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
import utils
from utils import reconcile
import dateutil.parser as parser
from transformers import AutoTokenizer, AutoModel
import torch
import os
from datetime import datetime
import ast
import io
from io import BytesIO

st.set_page_config(layout='wide')
lists=['site','subject','verbatim','decode','start_date','start_time',
       'end_date','end_time']
dtypes={"site":"string","subject":"string","verbatim":"string","decode":"string","start_date":"date","start_time":"time","end_date":"date","end_time":"time"}
merge_cols="subject,verbatim,decode,start_date,end_date"
df=pd.DataFrame()
df2=pd.DataFrame()
final2=pd.DataFrame()
study_src1_cols=''
study_src2_cols=''
lists2=lists

st.write('SAE Reconciliation')
ctn=st.container()
st.markdown('<style>.selectbox-class select {font-size: 10px;}</style>', unsafe_allow_html=True)
    
col1, col2 = ctn.columns(2,gap="small", vertical_alignment="top", border=True)
    
col1.write("**Clinical**")
uploaded_file = col1.file_uploader("Choose Clinical file in Excel format")

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.read())
    df = pd.read_excel(file_bytes)
    std_selected_options = col1.multiselect("Select standard one or more options:",
        lists)
    col1.text(std_selected_options)    
    selected_options = col1.multiselect("Select one or more options:",
        df.columns.tolist())
    
    col1.text(selected_options)       
    study_src1_cols=ast.literal_eval(str(dict(zip(std_selected_options,selected_options))))
    col1.write(df)

col2.write("**Safety**")
uploaded_file2 = col2.file_uploader("Choose Safety file in Excel format")
if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.read())
    df2= pd.read_excel(file_bytes)
    std_selected_options2 = col2.multiselect("Select standard one or more options 2:",
        lists2)
    col2.text(std_selected_options2)   
    selected_options2 = col2.multiselect("Select one or more options:",
        df2.columns.tolist())
    col2.text(selected_options2)
    study_src2_cols=ast.literal_eval(str(dict(zip(std_selected_options2,selected_options2))))
    col2.write(df2)

ctnx=st.container()
button1 = ctnx.button("Reconcile",on_click= reconcile, args= [ctnx,df,df2,study_src1_cols,study_src2_cols,dtypes,'clin_','pv_'], key="btn1")

