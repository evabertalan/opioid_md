import re
import numpy as np
import pandas as pd
import os, inspect

def evaluate_modeller(file_name, loop=False):
    log_file = '/Users/evabertalan/Dropbox/research_proj/code/modeller_script/loopmodel.log'
    model_score = []
    loop_score = []
    model_name = file_name + 'B'
    if loop:
        loop_name = file_name + 'BL'

    with open(log_file) as file:
        for i, line in enumerate(file):
            if loop and re.match(loop_name, line):
                line = line.split()
                loop_score.append(line)

            elif re.match(model_name, line):
                line = line.split()
                model_score.append(line)

    model_score = np.array(model_score)
    loop_score = np.array(loop_score)

    model_df = pd.DataFrame({
        'name': model_score[:, 0],
        'molpdf': np.array(model_score[:, 1], dtype=np.float32),
        'DOPE': np.array(model_score[:, 2], dtype=np.float32),
        'ga341': np.array(model_score[:, 3], dtype=np.float32),
        'norm_DOPE': np.array(model_score[:, 4], dtype=np.float32),
    })

    sorted_model = model_df.sort_values(by=['DOPE'], ascending=True)
    print(sorted_model)
    print('-----------------------------------------------------------------------')
    print('The best MODEL: ', sorted_model.iloc[0]['name'], ' ', sorted_model.iloc[0]['DOPE'])
    best_model = sorted_model.iloc[0]['name']

    if loop:
        loop_df = pd.DataFrame({
            'name': loop_score[:, 0],
            'molpdf': np.array(loop_score[:, 1], dtype=np.float32),
        })

        sorted_loop = loop_df.sort_values(by=['molpdf'], ascending=True)
        print(sorted_loop)
        print('-----------------------------------------------------------------------')
        print('The best LOOP model: ', sorted_loop.iloc[0]['name'], ' ', sorted_loop.iloc[0]['molpdf'])
        best_loop = sorted_loop.iloc[0]['name']

        return best_model, best_loop

    return best_model, None
