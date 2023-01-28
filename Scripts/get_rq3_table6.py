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
        if msg[0] in lemmatization:
            raise
        new_msgs.append(msg)
    for i in range(len(new_msgs)):
        new_msgs[i] = ' '.join(new_msgs[i])
    return new_msgs
def tongji_pattern(msgs, typee):

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
    for j, string in enumerate(msgs):
        flag = False
        for i in range(len(patterns)):
            if re.match(patterns[i], string):
                various_ids[i].append(j)
                flag = True
                break
        if flag == False:
            other_ids.append(j)
    return various_ids, other_ids
if __name__ == '__main__':
    
    for typee in ['add', 'remove', 'fix']:
        f = open('TablesAndFigures/rq3_table6_%s'%typee, 'w')
        raw_diffs = json.load(open('../DataSet/difftext.json'))
        raw_msgs = json.load(open('../DataSet/msg.json'))
        raw_msgs = preprocess(raw_msgs)

        all_index = json.load(open('../DataSet/all_index'))
        test_index = all_index['test']
        test_msgs = [raw_msgs[i] for i in test_index]
        test_diffs = [raw_diffs[i] for i in test_index]

        ref_various_ids, ref_othter_ids = tongji_pattern(test_msgs, 'test')
        test_ratio = sum([len(ref_various_ids[i]) for i in ref_various_ids]) / len(test_msgs)

    
        raw_marks = json.load(open('../DataSet/diffmark.json'))
        test_marks = [raw_marks[i] for i in test_index]
        add_mark_ids = []  
        for i in range(len(test_marks)):
            cur_test_mark = test_marks[i]
            add_num = len([each for each in cur_test_mark if each == 3])
            unchanged_num = len([each for each in cur_test_mark if each == 2])
            delete_num = len([each for each in cur_test_mark if each == 1])
            assert add_num + unchanged_num + delete_num == len(cur_test_mark)
            if typee == 'add':
                if add_num > 0 and delete_num == 0:
                    add_mark_ids.append(i)
            elif typee == 'remove':
                if delete_num > 0 and add_num == 0:
                    add_mark_ids.append(i)
            elif typee == 'fix':
                if delete_num > 0 and add_num > 0 :
                    add_mark_ids.append(i)

        mark_various_ids = {}
        mark_other_ids = []
        for i in ref_various_ids:
            mark_various_ids[i] = set(add_mark_ids) & set(ref_various_ids[i])
            cur_res = round(len(set(add_mark_ids) & set(ref_various_ids[i])) / len(add_mark_ids) * 100, 2)
            f.write('%s: %s\n'%(patterns[i].split(' ')[0], cur_res))
        cur_res = round(len(set(add_mark_ids) & set(ref_othter_ids)) / len(add_mark_ids) * 100, 2)
        mark_other_ids = set(add_mark_ids) & set(ref_othter_ids)
        f.write('other: %s\n'%(cur_res))


        for typee in ['ori', 'onlymark']:
            method_names = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
            total_method_pattern_ratios = {}
            total_method_pattern_bleus = {}
            total_method_pattern_bigbleu = {}
            total_method_other_ratios = []
            total_method_other_bleus = []
            total_method_other_bigbleu = []
            for method_name in method_names:
                if typee == 'ori':
                    method_msg_bleus = [x.strip() for x in open('../CommitMessages/EACH_BLEU/%s'%method_name) if x.strip()]
                elif typee == 'onlymark':
                    method_msg_bleus = [x.strip() for x in open('../CommitMessages/EACH_BLEU/%s_onlymark'%method_name) if x.strip()]
                method_msgs = [x.split(',')[0].split() for x in method_msg_bleus]
                method_bleus = [float(x.split(',')[1]) for x in method_msg_bleus]
                add_high_bleu_ids = []  
                for each in add_mark_ids:
                    if method_bleus[each] > 0.5:
                        add_high_bleu_ids.append(each)

                method_msgs = preprocess(method_msgs)
                method_various_ids, method_other_ids = tongji_pattern(method_msgs, method_name)
                f.write(method_name + '_%s\n'%typee)
                for i in method_various_ids:
                    if i not in total_method_pattern_ratios:
                        total_method_pattern_ratios[i] = []
                        total_method_pattern_bleus[i] = []
                        total_method_pattern_bigbleu[i] = []
                    cur_res  = round(len(set(add_mark_ids) & set(method_various_ids[i])) / len(add_mark_ids) * 100, 2)
                    cur_ids = mark_various_ids[i]
                    cur_bleus = [method_bleus[x] for x in cur_ids]
                    cur_avg_bleu = np.mean(cur_bleus) * 100

                    cur_big_bleus_ids = set(cur_ids) & set(add_high_bleu_ids)

                    f.write('%s: ratio %s bleu %s big_bleu_ratio %s\n'%(patterns[i].split(' ')[0], cur_res, round(cur_avg_bleu, 2), len(cur_big_bleus_ids) / len(add_high_bleu_ids)))
                    total_method_pattern_ratios[i].append(len(set(add_mark_ids) & set(method_various_ids[i])) / len(add_mark_ids) * 100)
                    total_method_pattern_bleus[i].append(cur_avg_bleu)
                    total_method_pattern_bigbleu[i].append(len(cur_big_bleus_ids) / len(add_high_bleu_ids))
                cur_res = round(len(set(add_mark_ids) & set(method_other_ids)) / len(add_mark_ids) * 100, 2)
                cur_ids = mark_other_ids
                cur_bleus = [method_bleus[x] for x in cur_ids]
                cur_avg_bleu = np.mean(cur_bleus) * 100
                cur_big_bleus_ids = set(cur_ids) & set(add_high_bleu_ids)
                f.write('other: %s bleu: %s big_bleu_ratio %s\n'%(cur_res, round(cur_avg_bleu, 2), len(cur_big_bleus_ids) / len(add_high_bleu_ids)))
                total_method_other_ratios.append(len(set(add_mark_ids) & set(method_other_ids)) / len(add_mark_ids) * 100)
                total_method_other_bleus.append(cur_avg_bleu)
                total_method_other_bigbleu.append(len(cur_big_bleus_ids) / len(add_high_bleu_ids))
            f.write('\ntotal:\n')
            for i in total_method_pattern_ratios:
                cur_res = round(np.mean(total_method_pattern_ratios[i]), 2)
                cur_bleu = round(np.mean(total_method_pattern_bleus[i]), 2)
                cur_bigbleu = round(np.mean(total_method_pattern_bigbleu[i]) * 100, 2)
                f.write('%s: ratio %s bleu %s bigbleu:%s\n'%(patterns[i].split(' ')[0], cur_res, cur_bleu, cur_bigbleu))
            cur_res = round(np.mean(total_method_other_ratios), 2)
            cur_bleu = round(np.mean(total_method_other_bleus), 2)
            cur_bigbleu = round(np.mean(total_method_other_bigbleu) * 100, 2)
            f.write('other: %s bleu %s bigbleu %s\n'%(cur_res, cur_bleu, cur_bigbleu))
            f.write('\n\n')
