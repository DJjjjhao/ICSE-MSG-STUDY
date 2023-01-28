
import re
import os 
import sys
import json
from nltk.stem import WordNetLemmatizer
import nltk.translate.bleu_score as bleu_score
import numpy as np
smooth_func = bleu_score.SmoothingFunction().method2

wnl = WordNetLemmatizer()

lemmatization = {"added": "add", "fixed": "fix", "removed": "remove", "adding": "add", "fixing": "fix", "removing": "remove"}

patterns = [r'remove (unused|unnecessary) .+', 
r'(fix .+ (in|to|of|when) .+)|(fix .+)', 
r'((don t|do not) .+)|((don t|do not) .+ if .+)',
r'(add .+ (for|to) .+)|(add .+)|(add missing .+ (for|to) .+)|(add missing .+)']

def preprocess(raw_msgs):
    # input:[[]]
    new_msgs = []
    for msg in raw_msgs:
        for i in range(len(msg)):
            msg[i] = msg[i].lower()
        msg[0] = wnl.lemmatize(msg[0], 'v')
        if msg[0] in lemmatization:
            raise
        new_msgs.append(msg)
    for i in range(len(new_msgs)):
        new_msgs[i] = ' '.join(new_msgs[i])
    return new_msgs
def tongji_pattern(msgs):

    total_num = len(msgs)

    various_ids = {}
    for i in range(len(patterns)):
        various_ids[i] = []
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
                various_ids[i].append(k)
                flag = True
                break
        if flag == False:
            other_ids.append(k)

    return various_ids, other_ids

def cal_pattern_other(models, pattern_ids, other_ids, msgs):
    results = {}
    for model in models:
        results[model] = {'pattern': {}, 'other': {}}
        for metric in metrics:
            metric_path = '../CommitMessages/EACH_%s/%s'%(metric, model)
            if metric == 'BLEU':
                cur_results = [float(x.strip().split(',')[1]) for x in open(metric_path) if x.strip()] 
            else:
                cur_results = eval(open(metric_path).read().strip())
            cur_results = [x * 100 for x in cur_results]
            # print(metric, len(cur_results), len(msgs))
            assert len(cur_results) == len(msgs)
            results[model]['pattern'][metric] = np.mean([cur_results[x] for x in pattern_ids])
            results[model]['other'][metric] = np.mean([cur_results[x] for x in other_ids])
    return results


def draw_result_table_v(file_name, caption, results_ori, results_mark):
    # vertical
    f = open(file_name, 'w')
    f.write('\\begin{table}[htbp]' + '\n')
    f.write('\\caption{%s}'%(caption) + '\n')
    f.write('\\begin{adjustbox}{width=\columnwidth}')
    symbol = 'ccc|' + 'rr|' * (len(metrics))
    f.write('\\begin{tabular}{%s}'%symbol + '\n')
    f.write('\\toprule' + '\n')
    f.write('\\multicolumn{3}{c|}{Model}')
    for i in range(len(metrics)):
        cur_metric = metrics[i] if metrics[i] != 'ROUGE' else 'ROUGE-L' 
        f.write('&')
        f.write('\\multicolumn{2}{c|}{%s/Decl}'%cur_metric)
    f.write('\\\\\n')
    for i in range(len(model_names)):
        f.write('\\midrule' + '\n')
        f.write('\\multirow{%s}{*}{%s}'%(2 * 2, tex_model_names[i])) 
        

        f.write('~&\\multirow{%s}{*}{%s}'%(2, 'Code\&Mark'))
        f.write('&Pattern')
        for metric in results_ori[models_ori[i]]['pattern']:
            f.write('&')
            f.write('%.2f'%results_ori[models_ori[i]]['pattern'][metric])
            f.write('&-')
        f.write('\\\\\n')
        f.write('~&~&Non-Pattern')
        for metric in results_ori[models_ori[i]]['other']:
            f.write('&')
            f.write('%.2f'%results_ori[models_ori[i]]['other'][metric])
            f.write('&-')
        f.write('\\\\\n')

        f.write('~&\\multirow{%s}{*}{%s}'%(2, 'Only Mark'))
        f.write('&Pattern')
        for metric in results_mark[models_mark[i]]['pattern']:
            f.write('&')
            cur_mark = results_mark[models_mark[i]]['pattern'][metric]
            cur_ori = results_ori[models_ori[i]]['pattern'][metric]
            f.write('%.2f'%(cur_mark))
            f.write('&')
            f.write('%.f\%%'%((cur_ori - cur_mark) / cur_ori * 100))
        f.write('\\\\\n')
        f.write('~&~&Non-Pattern')
        for metric in results_mark[models_mark[i]]['other']:
            f.write('&')
            cur_mark = results_mark[models_mark[i]]['other'][metric]
            cur_ori = results_ori[models_ori[i]]['other'][metric]
            f.write('%.2f'%(cur_mark))
            f.write('&')
            f.write('%.f\%%'%((cur_ori - cur_mark) / cur_ori * 100))
        f.write('\\\\\n')

    f.write('\\bottomrule' + '\n')
    f.write('\\end{tabular}' + '\n')
    f.write('\\end{adjustbox}' + '\n')
    f.write('\\end{table}' + '\n')
    f.close()

def cal_precision_recall(ref_ids, ids):
    recall = [0, 0]  
    for i in ref_ids:
        recall[1] += len(ref_ids[i])
        for j in ref_ids[i]:
            if j in ids[i]:
                recall[0] += 1
    recall = recall[0] / recall[1]
    precision = [0, 0]  
    for i in ids:
        precision[1] += len(ids[i])
        for j in ids[i]:
            if j in ref_ids[i]:
                precision[0] += 1
    precision = precision[0] / precision[1]
    return round(precision * 100, 2), round(recall * 100, 2)    

if __name__ == '__main__':
    raw_msgs = json.load(open('../DataSet/msg.json'))
    raw_msgs = preprocess(raw_msgs)

    all_index = json.load(open('../DataSet/all_index'))
    train_index = all_index['train']
    valid_index = all_index['valid']
    test_index = all_index['test']

    train_msgs = [raw_msgs[i] for i in train_index]
    valid_msgs = [raw_msgs[i] for i in valid_index]
    test_msgs = [raw_msgs[i] for i in test_index]

    various_ids_test, other_ids_test = tongji_pattern(test_msgs)
    pattern_ids_test = []
    for i in various_ids_test:
        pattern_ids_test += various_ids_test[i]

    model_names = ['NMT', 'PtrGNCMsg', 'CODISUM', 'CoreGen', 'FIRA']
    tex_model_names = ['\\nmt{}', '\\ptrgn{}', '\\codisum{}', '\\coregen{}', '\\fira{}']
    metrics = ['BLEU', 'ROUGE', 'METEOR']

    models_ori = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
    models_mark = ['nmt_onlymark', 'ptrgn_onlymark', 'codisum_onlymark', 'coregen_onlymark', 'fira_onlymark']
    results_mark = cal_pattern_other(models_mark, pattern_ids_test, other_ids_test, test_msgs)
    results_ori = cal_pattern_other(models_ori, pattern_ids_test, other_ids_test, test_msgs)


 
    file_name = 'TablesAndFigures/rq3_table7.tex'
    caption = '\\label{tab:repre_pattern_nopattern}Metrics on pattern group and non-pattern group of different input representation'
    draw_result_table_v(file_name, caption, results_ori, results_mark)