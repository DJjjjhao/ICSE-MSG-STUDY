import sys

method_name = sys.argv[1]
method_msgs = [x.strip().split(',')[0] for x in open('../CommitMessages/DefaultModels/output_%s'%method_name) if x.strip()]
with open('%s_spmf.text'%method_name, 'w') as f:
    for i in range(len(method_msgs)):
        if i != 0:
            f.write('. %s'%method_msgs[i])
        else:
            f.write(method_msgs[i])
            