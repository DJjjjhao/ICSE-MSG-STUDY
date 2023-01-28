import os
import time
import subprocess
import re
from tkinter import N
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

wnl = WordNetLemmatizer()

patterns = [r'remove (unused|unnecessary) .+', 
r'(fix .+ (in|to|of|when) .+)|(fix .+)', 
r'((don t|do not) .+)|((don t|do not) .+ if .+)',
r'(add .+ (for|to) .+)|(add .+)|(add missing .+ (for|to) .+)|(add missing .+)']



def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

def preprocess(raw_msgs):
    # input:[[]]
    new_msgs = []
    for msg in raw_msgs:
        for i in range(len(msg)):
            msg[i] = msg[i].lower()
        msg[0] = wnl.lemmatize(msg[0], 'v')
        new_msgs.append(msg)
    for i in range(len(new_msgs)):
        new_msgs[i] = ' '.join(new_msgs[i])
    return new_msgs

def tongji_pattern(msgs):

    various_ids = []
    other_ids = []

    # rename_strings = []
    # remove_strings = []
    # make_strings = []
    # add_strings = []
    # fix_strings = []
    # other_strings = []
    for k, string in enumerate(msgs):
        flag = False
        for i in range(len(patterns)):
            if re.match(patterns[i], string):
                various_ids.append(k)
                flag = True
                break
        if flag == False:
            other_ids.append(k)

    return various_ids, other_ids


if __name__ == '__main__':
    


    os.chdir('../CommitMessages')

    model_names = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
    output_names = {'nmt':'NMT', 'ptrgn':'PtrGNCMsg', 'codisum':'CODISUM', 'coregen':'CoreGen', 'fira':'FIRA'}

    execute_command('rm metrics_remove')
    for name in model_names:
        for i in range(10):
            ratio = i / 10 if i != 0 else 0
            execute_command('python B-Norm.py %s %s_%s ground_truth <  ModifiedDataset/output_%s_%s'%('metrics_remove', name, ratio, name, ratio))
    

    merge_metrics = [float(x.strip()) for x in open('metrics_remove') if x.strip()]
    print(len(merge_metrics))
    total_metrics = {}
    for name in model_names:
        total_metrics[name] = []

    for i in range(len(merge_metrics)):
        total_metrics[model_names[i // 10]].append(merge_metrics[i])


    total_pattern_metrics = {}
    total_nopattern_metrics = {}
    for name in model_names:
        total_pattern_metrics[name] = []
        total_nopattern_metrics[name] = []
    for i in range(10):
        ratio = i / 10 if i != 0 else 0
        msg_text = [x.strip().split() for x in open('ground_truth') if x.strip()]
        msg_text = preprocess(msg_text)
        pattern_ids, nopattern_ids = tongji_pattern(msg_text)
        for name in model_names:
            cur_path = 'EACH_BLEU/%s_%s'%(name, ratio)
            if not os.path.exists(cur_path):
                continue
            cur_bleus = [float(x.strip().split(',')[1]) * 100 for x in open(cur_path) if x.strip()]
            pattern_bleus = [cur_bleus[i] for i in pattern_ids]
            nopattern_bleus = [cur_bleus[i] for i in nopattern_ids]
            total_pattern_metrics[name].append(np.mean(pattern_bleus))
            total_nopattern_metrics[name].append(np.mean(nopattern_bleus))

    os.chdir('../Patterns')
    execute_command('rm patterns_remove')
    for name in model_names:
        for i in range(10):
            ratio = i / 10 if i != 0 else 0
            execute_command('python match_high_pattern.py %s_%s train 0 %s'%(name, ratio, 'patterns_remove'))
   
    merge_patterns = [float(x.strip()) for x in open('patterns_remove') if x.strip()]
    total_patterns = {}
    for name in model_names:
        total_patterns[name] = []

    for i in range(len(merge_patterns)):
        total_patterns[model_names[i // 10]].append(merge_patterns[i])


    CB91_Amber = '#F5B14C'    
    colors = ['b', 'g', 'c', CB91_Amber, 'tab:purple']

    os.chdir('../Scripts')
    # mpl.style.use('seaborn')
    plt.figure()
    for i, name in enumerate(model_names):
        ratio_decrease = [x * 10 for x in range(len(total_metrics[name]))]
        plt.plot(ratio_decrease, total_metrics[name], color=colors[i], label=output_names[name])  
    for i, name in enumerate(model_names):
        ratio_decrease = [x * 10 for x in range(len(total_metrics[name]))]
        plt.plot(ratio_decrease, total_pattern_metrics[name], color=colors[i], linestyle='--', label=output_names[name] + '-pattern', )  
    for i, name in enumerate(model_names):
        ratio_decrease = [x * 10 for x in range(len(total_metrics[name]))]
        plt.plot(ratio_decrease, total_nopattern_metrics[name], color=colors[i], linestyle=':', label=output_names[name] + '-non-pattern', )  

    plt.ylim(top=30)
    plt.xlabel('Ratio of decrease (%)')
    plt.ylabel('BLEU')
    plt.legend(fontsize=8.6, ncol =3,frameon=False)
    # plt.savefig('bleu_decrease.pdf')
    plt.savefig('TablesAndFigures/rq2_figure6.png')

    
    plt.figure()
    for i, name in enumerate(model_names):
        ratio_decrease = [x * 10 for x in range(len(total_patterns[name]))]

        plt.plot(ratio_decrease, total_patterns[name], color=colors[i], linewidth=2, label=output_names[name])  

    plt.ylim(top=100)
    plt.xlabel('Ratio of decrease (%)')
    plt.ylabel('Ratio of patterns (%)')
    plt.legend(fontsize=11,frameon=False)
    # plt.savefig('pattern_decrease.pdf')
    plt.savefig('TablesAndFigures/rq2_figure5.png')