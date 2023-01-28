
import os
import time
import subprocess

def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

if __name__ == '__main__':

    metrics = ['bleu', 'rouge', 'meteor']    
    model_names = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
    typees = ['', 'onlymark']

    os.chdir('../CommitMessages')
    
    execute_command('rm metrics_mark_code')
    
    for metric in metrics:
        for name in model_names:
            for typee in typees:
                if typee == '':
                    total_name = name
                    folder = 'DefaultModels'
                else:
                    total_name = name + '_' + typee
                    folder = 'InputRepresentation'
                if metric == 'bleu':
                    execute_command('python B-Norm.py %s %s ground_truth < %s/output_%s'%('metrics_mark_code', total_name, folder, total_name))
                elif metric == 'rouge':
                    execute_command('python Rouge.py --ref_path ground_truth  --gen_path %s/output_%s --file_name %s'%(folder, total_name, 'metrics_mark_code'))
                elif metric == 'meteor':
                    execute_command('python Meteor.py --ref_path ground_truth  --gen_path %s/output_%s --file_name %s'%(folder, total_name, 'metrics_mark_code'))


    os.chdir('../Patterns')
    execute_command('rm patterns_mark_code')
    for name in model_names:
        for typee in typees:
            if typee == '':
                total_name = name
            else:
                total_name = name + '_' + typee
            execute_command('python match_high_pattern.py %s test 0 %s'%(total_name, 'patterns_mark_code'))

    os.chdir('../Scripts')
    execute_command('python table_code_mark.py')


    