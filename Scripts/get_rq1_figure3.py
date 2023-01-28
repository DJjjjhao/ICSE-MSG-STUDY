import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import json
from numpy import average
from nltk.stem import WordNetLemmatizer
import nltk.translate.bleu_score as bleu_score
import re
import os
import numpy as np

wnl = WordNetLemmatizer()

patterns = [r'(add .+ (for|to) .+)|(add .+)|(add missing .+ (for|to) .+)|(add missing .+)',
r'remove (unused|unnecessary) .+', 
r'(fix .+ (in|to|of|when) .+)|(fix .+)', 
r'((don t|do not) .+)|((don t|do not) .+ if .+)']
pattern_name = ['Addition\nPattern', 'Removal\nPattern', 'Fix\nPattern', 'Avoidance\nPattern']
# pattern_name = ['Addition', 'Removal', 'Fix', 'Avoidance']
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

def get_lengths(method_msgs):
    
    lengths = {}
    lengths['All\nData'] = [] 
    num = 0
    for each in method_msgs:
        if len(each.split()) > 30:
            num += 1
            continue
        lengths['All\nData'].append(len(each.split()))
    print(num)
    various_strings, various_ids = tongji_pattern(method_msgs)
    lengths['All\nPatterns'] = []
    for i in various_ids:
        cur = pattern_name[i]
        lengths[cur] = []
        for j in various_ids[i]:
            if len(method_msgs[j].split()) > 30:
                continue
            lengths[cur].append(len(method_msgs[j].split()))
            lengths['All\nPatterns'].append(len(method_msgs[j].split()))

    return  lengths

def get_key_points(data):

    quartile1s = []
    quartile3s = []
    medians = []
    for each in data:
        quartile1, median, quartile3 = np.percentile(each, [25, 50, 75])
        quartile1s.append(quartile1)
        quartile3s.append(quartile3)
        medians.append(median)
    return quartile1s, medians, quartile3s



if __name__ == '__main__':
    model_names_train = ['ground_truth', 'nmt_train', 'ptrgn_train', 'codisum_train', 'coregen_train', 'fira_train']
    method_msgs = []
    for name in model_names_train:
        if name == 'ground_truth':
            raw_msgs = json.load(open('../DataSet/msg.json'))
            raw_msgs = preprocess(raw_msgs)

            all_index = json.load(open('../DataSet/all_index'))
            train_index = all_index['train']
            ref_msgs = [raw_msgs[i] for i in train_index]
        else:
            method_msg_bleus = [x.strip() for x in open('../CommitMessages/EACH_BLEU/%s'%name) if x.strip()]
            cur_msgs = [x.split(',')[0].split() for x in method_msg_bleus]
            method_msgs += preprocess(cur_msgs)
    ref_lengths = get_lengths(ref_msgs)
    method_lengths = get_lengths(method_msgs)

    '''
    final_lengths = []
    final_labels = []
    for i in ref_lengths:
        final_lengths.append(ref_lengths[i])
        final_labels.append(i)
    for i in method_lengths:
        final_lengths.append(method_lengths[i])
        final_labels.append(i)
        


    plt.boxplot(final_lengths, labels=final_labels) 
    plt.xlabel('Group')
    plt.xticks(fontsize=7)
    plt.ylabel('Length')
    # plt.boxplot((**all_bleus))
    # plt.ylim(top=30)
    plt.show()
    plt.savefig('lengths.png')
    plt.savefig('lengths.pdf')
    '''


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.subplots_adjust(wspace=0.05)
    # rectangular box plot
    bplot1 = ax1.boxplot(list(ref_lengths.values()),
                        # notch=True,  # notch shape
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=list(ref_lengths.keys()), showfliers=False)  # will be used to label x-ticks
    ax1.set_title('Ground Truth Commit Messages')

    # notch shape box plot
    bplot2 = ax2.boxplot(list(method_lengths.values()),
                        
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=list(method_lengths.keys()), showfliers=False)  # will be used to label x-ticks
    ax2.set_title('Generated Commit Messages')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2):
        for i in range(len(bplot['boxes'])):
            bplot['boxes'][i].set_facecolor(colors[min(2, i)])

    # adding horizontal grid lines
    for i, ax in enumerate([ax1, ax2]):
        # ax.set_ylim(top=20)
        ax.set_yticks(np.arange(0, 21, 4))
        ax.yaxis.grid(True)
        # ax.set_xlabel('Group',fontsize=7.5)
        if i == 0:
            ax.set_ylabel('Length')
        else:
            ax.set_yticklabels([])
        # ax.tick_params(axis='x', labelsize=8.5)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.tick_params(axis='y', labelsize=9.5)
    # fig.align_ylabels([ax1, ax2])
    

    plt.show()
    plt.savefig('TablesAndFigures/rq1_figure3.png')
    # plt.savefig('TablesAndFigures/lengths.pdf')


    data_ref = list(ref_lengths.values())
    data_method = list(method_lengths.values())


    quartile1_ref, medians_ref, quartile3_ref = get_key_points(data_ref)
    quartile1_method, medians_method, quartile3_method = get_key_points(data_method)
    
    quartile1_dec = [quartile1_ref[i] - quartile1_method[i] for i in range(len(quartile1_ref))]
    medians_dec = [medians_ref[i] - medians_method[i] for i in range(len(medians_method))]
    quartile3_dec = [quartile3_ref[i] - quartile3_method[i] for i in range(len(quartile3_method))]
    # with open('length_decrease_between_ref_generated', 'w') as f:
    #     f.write('quartile1_dec: %s\n'%quartile1_dec)
    #     f.write('medians_dec: %s\n'%medians_dec)
    #     f.write('quartile3_dec: %s\n'%quartile3_dec)
    #     f.write('avg of ground truth:\n')
    #     for each in ref_lengths:
    #         f.write('%s: %s\n'%(each, np.mean(ref_lengths[each])))
    #     f.write('avg of generated:\n')
    #     for each in method_lengths:
    #         f.write('%s: %s\n'%(each, np.mean(method_lengths[each])))
            