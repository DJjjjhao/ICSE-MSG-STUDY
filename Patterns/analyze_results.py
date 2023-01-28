import os
import sys
import json
if __name__ == '__main__':
    file_name = sys.argv[1]
    typee = sys.argv[2]
    threshold = float(sys.argv[3])
    all_index = json.load(open('../DataSet/all_index'))
    total_num = len(all_index[typee])
    raw_results = [x.strip() for x in open(file_name) if x.strip()]
    results = {}
    for each in raw_results:
        text, occurs = each.split('#SUP:')
        text = text.strip()
        occurs = int(occurs.strip()) / total_num
        if occurs < threshold:
            continue
        text = text.split('-1')
        text = [x.strip() for x in text if x and x.strip()]
        if len(text) == 1:
            continue
        text = ' '.join(text)
        results[text] = occurs
    results = dict(sorted(results.items(), key=lambda item: (item[0].split()[0], item[0].split()[1], item[1]), reverse=True))
    # results = dict(sorted(results.items(), key=lambda item: (item[0].split()[0], item[1]), reverse=True))
    # results = dict(sorted(results.items(), key=lambda item: (item[1]), reverse=True))
    
    total = 0
    for text in results:
        # print(results[text])
        total += results[text]
    results['total'] = total
    json.dump(results, open('results/%s_patterns'%(file_name.split('/')[1]), 'w'), indent=0, ensure_ascii=False)

        