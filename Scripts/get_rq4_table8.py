
import os
import time
import subprocess

def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

if __name__ == '__main__':

    model_names = [ 'nmt', 'nmt_noatten', 
                    'ptrgn', 'ptrgn_noatten', 'ptrgn_nocopy', 
                    'codisum', 'codisum_noatten', 'codisum_nocopy', 'codisum_noanony',
                    'fira', 'fira_nocopy', 'fira_noanony'
                    ]
    os.chdir('../CommitMessages')

    save_path = 'metrics_patterns_componant'
    execute_command('rm ../Scripts/' + save_path)
    
    for total_name in model_names:
        if total_name in ['nmt', 'ptrgn', 'codisum', 'fira']:
            folder = 'DefaultModels'
        else:
            folder = 'Component'
        os.chdir('../CommitMessages')
        execute_command('python B-Norm.py ../Scripts/%s %s ground_truth < %s/output_%s'%(save_path, total_name, folder, total_name))
        execute_command('python Rouge.py --ref_path ground_truth  --gen_path %s/output_%s --file_name ../Scripts/%s'%(folder, total_name, save_path))
        execute_command('python Meteor.py --ref_path ground_truth  --gen_path %s/output_%s --file_name ../Scripts/%s'%(folder, total_name, save_path))
        os.chdir('../Patterns')
        execute_command('python match_high_pattern.py %s test 0 ../Scripts/%s'%(total_name, save_path))

    os.chdir('../Scripts')
    execute_command('rm ' + 'TablesAndFigures/rq4_table6.tex')
    execute_command('python table_line.py %s %s %s'%(save_path, 'TablesAndFigures/rq4_table6.tex', 4))



    