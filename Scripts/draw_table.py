
import sys
import json
from numpy import average
from nltk.stem import WordNetLemmatizer
import nltk.translate.bleu_score as bleu_score
import re
import os
import numpy as np

wnl = WordNetLemmatizer()

patterns = [
r'(add .+ (for|to) .+)|(add .+)|(add missing .+ (for|to) .+)|(add missing .+)',
r'remove (unused|unnecessary) .+', 
r'(fix .+ (in|to|of|when) .+)|(fix .+)', 
r'((don t|do not) .+)|((don t|do not) .+ if .+)'
]


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

    various_strings = {}
    various_ids = {}
    for i in range(len(patterns)):
        various_strings[i] = []
        various_ids[i] = []
    other_strings = []

    for j, string in enumerate(msgs):
        flag = False
        for i in range(len(patterns)):
            if re.match(patterns[i], string):
                various_strings[i].append(string)
                various_ids[i].append(j)
                flag = True
                break
        if flag == False:
            other_strings.append(string)

    return various_strings, various_ids

def get_lengths():
    raw_msgs = json.load(open('../DataSet/msg.json'))
    raw_msgs = preprocess(raw_msgs)

    all_index = json.load(open('../DataSet/all_index'))
    train_index = all_index['train']
    valid_index = all_index['valid']
    test_index = all_index['test']

    train_msgs = [raw_msgs[i] for i in train_index]
    valid_msgs = [raw_msgs[i] for i in valid_index]
    test_msgs = [raw_msgs[i] for i in test_index]
    
    various_strings, various_ids = tongji_pattern(train_msgs)

    length_patterns_train = {}
    for i in various_ids:
        length_patterns_train[i] = []
        for j in various_ids[i]:
            length_patterns_train[i].append(len(train_msgs[j].split()))
    average_length_train = []
    average_length_train.append(average(np.concatenate(list(length_patterns_train.values()))))
    for i in length_patterns_train:
        average_length_train.append(average(length_patterns_train[i]))
    
    various_strings, various_ids = tongji_pattern(test_msgs)

    length_patterns_test = {}
    for i in various_ids:
        length_patterns_test[i] = []
        for j in various_ids[i]:
            length_patterns_test[i].append(len(test_msgs[j].split()))
    average_length_test = []
    average_length_test.append(average(np.concatenate(list(length_patterns_test.values()))))
    for i in length_patterns_test:
        average_length_test.append(average(length_patterns_test[i]))
    
    for i in range(len(average_length_train)):
        average_length_train[i] = round(average_length_train[i], 2)
    for i in range(len(average_length_test)):
        average_length_test[i] = round(average_length_test[i], 2)
    return average_length_train, average_length_test 


def draw_result_table_v(file_name, caption, headers, cur_results_train, cur_results_test):
    # vertical
    f = open(file_name, 'w')
    f.write('\\begin{table}[htbp]' + '\n')
    f.write('\\caption{%s}'%(caption) + '\n')
    f.write('\\begin{adjustbox}{width=\\columnwidth}' + '\n')
    symbol = 'cc|' + 'c' * (len(model_names))
    # symbol = 'c' * (len(model_names) + 2)
    f.write('\\begin{tabular}{%s}'%symbol + '\n')
    f.write('\\toprule' + '\n')
    f.write('\\multicolumn{2}{c|}{Model}')
    for i in range(len(model_names)):
        f.write('&')
        f.write(tex_model_names[model_names[i]] if model_names[i] in tex_model_names else model_names[i])
    f.write('\\\\\n')
    f.write('\\midrule' + '\n')
    f.write('\\multirow{%s}{*}{Training Set}'%len(headers))
    for i in range(len(cur_results_train[0])):
        if i != 0:
            f.write('~')
        f.write('&%s'%headers[i])
        for j in range(len(cur_results_train)):
            f.write('&%.2f'%cur_results_train[j][i])
        f.write('\\\\\n')
    f.write('\\midrule' + '\n') 
    f.write('\\multirow{%s}{*}{Testing Set}'%len(headers))
    for i in range(len(cur_results_test[0])):
        if i != 0:
            f.write('~')
        f.write('&%s'%headers[i])
        for j in range(len(cur_results_test)):
            f.write('&%.2f'%cur_results_test[j][i])
        f.write('\\\\\n')
    
    
    f.write('\\bottomrule' + '\n')
    f.write('\\end{tabular}' + '\n')
    f.write('\\end{adjustbox}' + '\n')
    f.write('\\end{table}' + '\n')
    f.close()


if __name__ == '__main__':
    metric_file_name = 'TablesAndFigures/rq1_table1.tex'
    pattern_file_name = 'TablesAndFigures/rq1_table2.tex'


    model_names = ['NMT', 'PtrGNCMsg', 'CODISUM', 'CoreGen', 'FIRA']
    tex_model_names = {'NMT':'\\nmt{}', 'PtrGNCMsg':'\\ptrgn{}', 'CODISUM':'\\codisum{}', 'CoreGen':'\\coregen{}', 'FIRA':'\\fira{}'}
    metric_headers = ['BLEU', 'ROUGE-L', 'METEOR']
    num_patterns = 4
    pattern_headers = ['All Patterns', 'Addition','Removal', 'Fix', 'Avoidance']

    results_train = [x.strip().split() for x in open('results_train') if x.strip()]  
    results_test = [x.strip().split() for x in open('results_test') if x.strip()]
    
    for i in range(len(results_train)):
        for j in range(len(results_train[i])):
            results_train[i][j] = float(results_train[i][j])
            results_test[i][j] = float(results_test[i][j])
            if j >= len(metric_headers):
                results_train[i][j] = results_train[i][j] * 100
                results_test[i][j] = results_test[i][j] * 100
    assert len(results_train) == len(results_test) == len(model_names)
    
    # metric table
    metric_caption = "\\label{tab:metrics}NLP metrics of various techniques. It's worth noting that the data in all tables of this paper are omitted \\%."
    metric_results_train = []
    metric_results_test = []
    for result in results_train:
        metric_results_train.append(result[:len(metric_headers)])
    for result in results_test:
        metric_results_test.append(result[:len(metric_headers)])
    draw_result_table_v(metric_file_name, metric_caption, metric_headers, metric_results_train, metric_results_test)

    # pattern table
    
    model_names = ['Ground Truth', 'NMT', 'PtrGNCMsg', 'CODISUM', 'CoreGen', 'FIRA', 'Average Increase']
    tex_model_names = {'NMT':'\\nmt{}', 'PtrGNCMsg':'\\ptrgn{}', 'CODISUM':'\\codisum{}', 'CoreGen':'\\coregen{}', 'FIRA':'\\fira{}'}

    patterns_data_train  = [x.strip().split() for x in open('../Patterns/pattern_ground_truth_train') if x.strip()][0]
    patterns_data_train = [float(x) * 100 for x in patterns_data_train]
    patterns_data_test  = [x.strip().split() for x in open('../Patterns/pattern_ground_truth_test') if x.strip()][0]
    patterns_data_test = [float(x) * 100 for x in patterns_data_test]


    average_length_train, average_length_test = get_lengths()
    pattern_caption = '\\label{tab:patterns}Pattern ratios of various techniques'
    pattern_results_train = []
    pattern_results_test = []
    pattern_results_train.append(patterns_data_train)
    for result in results_train:
        pattern_results_train.append(result[len(metric_headers):])
    # add the average increase
    average_increase_train = []
    for i in range(len(pattern_results_train[0])):
        cur_increase = []
        for j in range(1, len(pattern_results_train)):
            cur_increase.append(pattern_results_train[j][i] - pattern_results_train[0][i])
        average_increase_train.append(sum(cur_increase) / len(cur_increase))
    pattern_results_train.append(average_increase_train)
    # pattern_results_train.append(average_length_train)

    pattern_results_test.append(patterns_data_test)
    for result in results_test:
        pattern_results_test.append(result[len(metric_headers):])

    # add the average increase
    average_increase_test = []
    for i in range(len(pattern_results_test[0])):
        cur_increase = []
        for j in range(1, len(pattern_results_test)):
            cur_increase.append(pattern_results_test[j][i] - pattern_results_test[0][i])
        average_increase_test.append(sum(cur_increase) / len(cur_increase))
    pattern_results_test.append(average_increase_test)
    # pattern_results_test.append(average_length_test)

    draw_result_table_v(pattern_file_name, pattern_caption, pattern_headers, pattern_results_train, pattern_results_test)