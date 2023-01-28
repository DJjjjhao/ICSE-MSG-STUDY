import sys
if __name__ == '__main__':
    stage = sys.argv[1]
    model_names = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
    total_results = []
    for i in range(len(model_names)):
        total_results.append([])
    typees = ['bleu', 'rouge', 'meteor']
    result = [x.strip().split() for x in open('../CommitMessages/metrics_study_%s'%stage) if x.strip()][0]

    for i in range(len(typees)):
        for j in range(len(model_names)):
            index = i * len(model_names) + j
            total_results[j].append(result[index])
    
    # for typee in typees:
    #     result = [x.strip().split() for x in open('../../RESULT_ICSE_STUDY/CommitMessages/%s_study_test'%typee) if x.strip()][0]
    #     assert len(result) == len(model_names)
    #     for i in range(len(result)):
    #         total_results[i].append(result[i])
    result = [x.strip().split() for x in open('../Patterns/pattern_study_%s'%stage) if x.strip()]
    assert len(result) == len(model_names)
    for i in range(len(result)):
        total_results[i] += result[i]
    f = open('results_%s'%stage, 'w')
    for result in total_results:
        f.write(' '.join(result) + '\n')