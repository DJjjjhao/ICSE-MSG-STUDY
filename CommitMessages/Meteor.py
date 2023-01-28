import os,argparse
import numpy as np
from nltk.translate.meteor_score import meteor_score

# import nltk
# nltk.download('wordnet')

def get_meteor(ref_path, gen_path, is_sentence=False):
    gen_sentence_lst = open(gen_path).read().split("\n")
    ref_sentence_lst = open(ref_path).read().split("\n")
    sentence_bleu_lst = [meteor_score([ref_sentence], gen_sentence) for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    f.write(str(sentence_bleu_lst))
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu*100

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate Meteor by NLTK')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)
    parser.add_argument("-f", "--file_name", required = False)

    args = parser.parse_args()

    method_name = (args.gen_path).split('/')[1].lstrip('output_')
    if method_name.startswith('rgn'):
        method_name = 'pt' + method_name
    f = open('EACH_METEOR/%s'%method_name, 'w')
    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        result = get_meteor(args.ref_path, args.gen_path)
        print(result)
        if args.file_name:
            open(args.file_name, 'a').write(str(result) + '\n')
  
    else:
        print("File not exits")