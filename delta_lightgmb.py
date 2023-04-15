#!/usr/bin/env python

import sys
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
from tqdm.auto import tqdm
from glob import glob

def create_delta_df(df_in, activity_column ='pIC50', method = "concat"):
    merge_df = pd.merge(df_in,df_in,how='cross')
    merge_df['delta'] = merge_df[activity_column+"_x"] - merge_df[activity_column+"_y"]
    if method == "subtract":
        merge_df['combo_fp'] = [a-b for a,b in merge_df[["fp_x","fp_y"]].values]
    elif method == 'concat':
        merge_df['combo_fp'] = [np.concatenate([a,b]) for a,b in merge_df[["fp_x","fp_y"]].values]
    else:
        assert False,f"create_delta_df {method} is not a vaild choice"
    return merge_df


def run_test(df_in, num_cycles, model=LGBMRegressor, n_jobs=-1):
    calc = FPCalculator("ecfp")
    trans = MoleculeTransformer(calc)
    df_in['fp'] = trans.transform(df_in.SMILES.values)
    output_list = []
    for i in tqdm(range(0,num_cycles)):
        train, test = train_test_split(df_in)
        
        single_model = model(n_jobs=n_jobs)
        single_model.fit(np.stack(train.fp),train.pIC50)
        single_pred = single_model.predict(np.stack(test.fp))
        tmp_df = test.copy()
        tmp_df['pred'] = single_pred
        tmp_merge_df = create_delta_df(tmp_df)
        tmp_merge_df['pred_delta'] = tmp_merge_df.pred_x - tmp_merge_df.pred_y
        r2_single = r2_score(tmp_merge_df.delta, tmp_merge_df.pred_delta)
        rmse_single = mean_squared_error(tmp_merge_df.delta, tmp_merge_df.pred_delta, squared=False)

        
        train_delta_df = create_delta_df(train, method='concat')
        test_delta_df = create_delta_df(test, method='concat')
        delta_model = model(n_jobs=n_jobs)
        delta_model.fit(np.stack(train_delta_df.combo_fp), train_delta_df.delta)
        delta_pred = delta_model.predict(np.stack(test_delta_df.combo_fp))
        r2_delta = r2_score(test_delta_df.delta, delta_pred)
        rmse_delta = mean_squared_error(test_delta_df.delta, delta_pred, squared=False)

        output_list.append([i, r2_single, r2_delta, rmse_single, rmse_delta])
    return output_list

def process_file(filename,size_cutoff=1500):
    df = pd.read_csv(filename,sep=" ",names=["SMILES","Name","pIC50"])
    res_df = None
    if len(df) < size_cutoff:
        print(filename)
        res = run_test(df,10)
        res_df = pd.DataFrame(res,columns=["cycle","r2_single","r2_delta","rmse_single","rmse_delta"])
        res_df['filename'] = filename
    return res_df

def main():
    # On my machine I run out of memory around 1600 molecules
    max_mols = 1600
    df_list = []
    for filename in sorted(glob("*/*.smi")):
        res_df = process_file(filename,size_cutoff=max_mols)
        if res_df is not None:
            df_list.append(res_df)
    combo_df = pd.concat(df_list)
    combo_df.to_csv("delta_data.csv",index=False)
    
if __name__ == "__main__":
    main()
    


