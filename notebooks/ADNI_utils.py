import numpy as np
import pandas as pd
import os


# print term definitions and codes, and 
def define_terms(data_df, dict_df, table_name=None, columns=None):
    
    if columns is None:
        columns = data_df.columns
    if table_name is None:
        keys = ["FLDNAME","TYPE","TEXT","CODE"]
    else:
        keys = ["FLDNAME","TYPE","TBLNAME","TEXT","CODE"]
        
    term_dicts = []
    for col in columns:

        term_dict = dict.fromkeys(keys)
        if table_name is None:
            loc = (dict_df.FLDNAME == col)
        else:
            loc = (dict_df.FLDNAME == col) & (dict_df.TBLNAME == table_name)
        tmp = dict_df.loc[loc][keys]
        
        for key in keys:
            if tmp[key].unique().shape[0]:
                term_dict[key] = tmp[key].unique()[0]
            else:
                term_dict[key] = float('nan')
        
        term_dicts.append(term_dict)
    
    data_dict = pd.DataFrame.from_dict(term_dicts).reindex(columns=keys)
    return(data_dict)



# function that searches recursively under an input directory for all files of a given extension
def paths_with_ext(directory=None, extension=".csv"):

    if directory is None:
        directory = os.getcwd()

    matches=[]
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                matches.append(os.path.join(root, filename))
                
    return(matches)


# report num entries, columns, patients, and range of num records and phases covered
def describe_meta_data(df):

    by_patient = df.groupby("RID")
    nRecords = by_patient.apply(len)
    nPatients = nRecords.shape[0]
    if "Phase" in df.columns:
        nPhases = by_patient.apply(lambda x: len(x.Phase.unique()))
        print("Phases:\t {}".format(df.Phase.unique()))
    else:
        nPhases = 0
        print("No phases listed")
    duplicate_in_phase  = np.sum(nRecords != nPhases)    
    
    print("Num Entries: %i\nNum Columns: %i" % (df.shape[0],df.shape[1]))
    print("Num Patients: %i\nRecords per Patient: %i-%i" % (nPatients,np.min(nRecords),np.max(nRecords)))
    print("Phases spanned per patient: %i-%i" % (np.min(nPhases),np.max(nPhases)))
    print("Patients w/ Duplicates: %i" % duplicate_in_phase)



# add new patient_data from  by-patient dataframe
def combine_patient_data(pat_df, new_df):
    
    # intialize empty placeholder columns in patient data
    new_cols = new_df.columns
    
    # remove duplicate columns
    new_cols = list(set(new_cols)-set(pat_df.columns))
    pat_df = pat_df.reindex(columns=pat_df.columns.tolist() + new_cols)
    
    # compare list of patient IDs
    old_RIDs = pat_df.index
    new_RIDs = new_df.index
    is_old= np.in1d(new_RIDs,old_RIDs)
    
    # grab data from existing patient list to insert into pat_df
    insert_mat = new_df.loc[is_old, new_cols].values
    
    #import pdb;pdb.set_trace()
    pat_df.loc[new_RIDs[is_old],new_cols] = insert_mat
    
    # append new patient data from RIDs non-existent in pat_df
    pat_df = pat_df.append(new_df.loc[~is_old], sort=True)
    
    return(pat_df)