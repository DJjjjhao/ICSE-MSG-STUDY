import argparse
import json
import os
import numpy as np

### Rouge
def get_rouge(ref_path, gen_path, is_sentence=False):
    if is_sentence:
        evaluate_cmd = "sumeval r-nl \"{}\" \"{}\"".format(gen_path, ref_path)
    else:
        evaluate_cmd = "sumeval r-nl -f \"{}\" \"{}\" -in".format(gen_path, ref_path)
    # print(json.load(os.popen(evaluate_cmd)))
    rouge_score = json.load(os.popen(evaluate_cmd))["averages"]
    scores = json.load(os.popen(evaluate_cmd))["scores"]
    scores = [x['ROUGE-L'] for x in scores]
    # print(np.mean(scores), rouge_score["ROUGE-L"])
    f.write(str(scores))
    for key in rouge_score.keys():
        rouge_score[key] = 100*rouge_score[key]
    return rouge_score

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate Rouge-1, Rouge-2, Rouge-N by sumeval')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)
    parser.add_argument("-f", "--file_name", required = False)

    args = parser.parse_args()

    method_name = (args.gen_path).split('/')[1].lstrip('output_')
    if method_name.startswith('rgn'):
        method_name = 'pt' + method_name
    f = open('EACH_ROUGE/%s'%method_name, 'w')
    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        result = get_rouge(args.ref_path, args.gen_path)['ROUGE-L']
        print(result)
        if args.file_name:
            open(args.file_name, 'a').write(str(result) + '\n')

    else:
        print("File not exits")