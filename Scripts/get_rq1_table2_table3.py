import os
import time
import subprocess
def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

patterns = [
r'(add .+ (for|to) .+)|(add .+)|(add missing .+ (for|to) .+)|(add missing .+)',
r'remove (unused|unnecessary) .+', 
r'(fix .+ (in|to|of|when) .+)|(fix .+)', 
r'((don t|do not) .+)|((don t|do not) .+ if .+)'
]


if __name__ == '__main__':
    
    model_names_train = ['nmt_train', 'ptrgn_train', 'codisum_train', 'coregen_train', 'fira_train']
    model_names_test = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']

    os.chdir('../CommitMessages')

    execute_command('rm metrics_study_train')
    for name in model_names_train:
        execute_command('python B-Norm-study.py train %s ground_truth_train <  DefaultModels/output_%s'%(name, name))
    for name in model_names_train:
        execute_command('python Rouge-study.py --ref_path ground_truth_train  --gen_path DefaultModels/output_%s --stage train'%name)
    for name in model_names_train:
        execute_command('python Meteor-study.py --ref_path ground_truth_train  --gen_path DefaultModels/output_%s --stage train'%name)
    
    execute_command('rm metrics_study_test')
    for name in model_names_test:
        execute_command('python B-Norm-study.py test %s ground_truth <  DefaultModels/output_%s'%(name, name))
    for name in model_names_test:
        execute_command('python Rouge-study.py --ref_path ground_truth  --gen_path DefaultModels/output_%s --stage test'%name)
    for name in model_names_test:
        execute_command('python Meteor-study.py --ref_path ground_truth  --gen_path DefaultModels/output_%s --stage test'%name)

    os.chdir('../Patterns')
    execute_command('rm pattern_study_train')
    for name in model_names_train:
        execute_command('python match_high_pattern_drawtable.py %s train 0'%name)
    execute_command('rm pattern_study_test')
    for name in model_names_test:
        execute_command('python match_high_pattern_drawtable.py %s test 0'%name)
    
    os.chdir('../Scripts')
    
    execute_command('python merge_results.py train')
    execute_command('python merge_results.py test')
    execute_command('python draw_table.py')

    # output => 'table1.tex' 'table2.tex'


    