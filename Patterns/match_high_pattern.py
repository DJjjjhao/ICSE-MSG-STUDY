import re
import os 
import sys
import json
from nltk.stem import WordNetLemmatizer
import nltk.translate.bleu_score as bleu_score
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
def tongji_pattern(msgs, typee):

    total_num = len(msgs)

    various_strings = {}
    for i in range(len(patterns)):
        various_strings[i] = []
    other_strings = []

    # rename_strings = []
    # remove_strings = []
    # make_strings = []
    # add_strings = []
    # fix_strings = []
    # other_strings = []
    for string in msgs:
        flag = False
        for i in range(len(patterns)):
            if re.match(patterns[i], string):
                various_strings[i].append(string)
                flag = True
                break
        if flag == False:
            other_strings.append(string)

    with open(method_name + '/%s_pattern'%typee, 'w') as f:
        for i in various_strings:
            for string in various_strings[i]:
                f.write(string + '\n')
    with open(method_name + '/%s_other'%typee, 'w') as f:
        for string in other_strings:
            f.write(string + '\n')
    return various_strings
if __name__ == '__main__':
    method_name = sys.argv[1]
    stage = sys.argv[2]
    # train, test
    reduced = bool(int(sys.argv[3]))
    # print(reduced)
    # patterns = all_patterns[method_name.split('_')[0]]
    record_file = sys.argv[4]



    if not os.path.exists(method_name):
        os.makedirs(method_name)
    p = open(method_name + '/ratios', 'w')
    raw_msgs = json.load(open('../DataSet/msg.json'))
    raw_msgs = preprocess(raw_msgs)

    all_index = json.load(open('../DataSet/all_index'))
    train_index = all_index['train']
    valid_index = all_index['valid']
    test_index = all_index['test']



    train_msgs = [raw_msgs[i] for i in train_index]
    valid_msgs = [raw_msgs[i] for i in valid_index]
    test_msgs = [raw_msgs[i] for i in test_index]
    
    various_strings = tongji_pattern(train_msgs, 'train')
    train_ratio = sum([len(various_strings[i]) for i in various_strings]) / len(train_msgs)
    p.write('train ratio:%s\n'%train_ratio)

  

    various_strings = tongji_pattern(valid_msgs, 'valid')
    valid_ratio = sum([len(various_strings[i]) for i in various_strings]) / len(valid_msgs)
    p.write('valid ratio:%s\n'%valid_ratio)

    various_strings = tongji_pattern(test_msgs, 'test')
    test_ratio = sum([len(various_strings[i]) for i in various_strings]) / len(test_msgs)
    p.write('test ratio:%s\n'%test_ratio)
    


    method_msg_bleus = [x.strip() for x in open('../CommitMessages/EACH_BLEU/%s'%method_name) if x.strip()]
    method_msgs = [x.split(',')[0].split() for x in method_msg_bleus]
    method_bleus = [float(x.split(',')[1]) for x in method_msg_bleus]

    method_msgs = preprocess(method_msgs)
    

    
    various_strings = tongji_pattern(method_msgs, method_name)
    method_ratio = sum([len(various_strings[i]) for i in various_strings]) / len(method_msgs)
    # ff.write(str(method_ratio) + ' ')

    p.write('%s ratio:%s\n'%(method_name, method_ratio))
    print("%.2f"%(method_ratio * 100))
    if record_file.lower != 'none':
        open(record_file, 'a').write("%.2f"%(method_ratio * 100) + '\n')


    for i in various_strings:
        p.write('%s:%s\n'%(patterns[i], len(various_strings[i]) / len(method_msgs)))
        # ff.write(str(len(various_strings[i]) / len(method_msgs)) + ' ')
    # ff.write('\n')
    # ff.close()

    # if method_name.endswith('train'):
    #     for i in range(len(method_msgs)):
    #         method_bleus.append(bleu_score.sentence_bleu([train_msgs[i].split()], method_msgs[i].split(), smoothing_function=smooth_func))
    # else:
    #     for i in range(len(method_msgs)):
    #         method_bleus.append(bleu_score.sentence_bleu([test_msgs[i].split()], method_msgs[i].split(), smoothing_function=smooth_func))


    
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratios = []
    for threshold in thresholds:
        cur_msgs = []
        for i in range(len(method_bleus)):
            if method_bleus[i] >= threshold:
                cur_msgs.append(method_msgs[i])
        # print(cur_msgs)
        if len(cur_msgs) == 0:
            continue
        various_strings = tongji_pattern(cur_msgs, '%s_%s'%(method_name, threshold))
        cur_ratio = sum([len(various_strings[i]) for i in various_strings]) / len(cur_msgs)
        p.write('%s_%s ratio:%s\n'%(method_name, threshold, cur_ratio))
        ratios.append(cur_ratio)
    
    open('pattern_bleu_%s_%s'%(method_name.split('_')[0], stage), 'w').write(str(ratios))