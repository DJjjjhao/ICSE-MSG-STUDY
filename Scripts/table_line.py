import sys

data_path = sys.argv[1]
output_path = sys.argv[2]
num = int(sys.argv[3])  
data = [float(x.strip()) for x in open(data_path) if x.strip()]
model_names = ['NMT', 'PtrGNCMsg', 'CODISUM', 'FIRA']
metrics = ['BLEU', 'ROUGE-L', 'METEOR']
all_data = {}
typees = {}
for model_name in model_names:
    all_data[model_name] = []
typees['NMT'] = ['Original', 'Attention mechanism']
typees['PtrGNCMsg'] = ['Original', 'Attention mechanism', 'Copy mechanism']
typees['CODISUM'] = ['Original', 'Attention mechanism', 'Copy mechanism', 'Anonymization']
typees['FIRA'] = ['Original', 'Copy mechanism', 'Anonymization']
for i in range(len(data) // num):
    cur_data = data[i * num:(i + 1) * num]
    if i <= 1:
        all_data['NMT'].append(cur_data)
    elif i <= 4:
        all_data['PtrGNCMsg'].append(cur_data)
    elif i <= 8:
        all_data['CODISUM'].append(cur_data)
    else:
        all_data['FIRA'].append(cur_data)
f = open(output_path, 'w')
f.write('\\begin{table}[htbp]' + '\n')
f.write('\\caption{%s}'%('Metrics and pattern ratio of model removing each component') + '\n')


symbol = 'cc|' + 'c' * (len(metrics) + 1)
f.write('\\begin{tabular}{%s}'%symbol + '\n')
f.write('\\toprule' + '\n')
f.write('\\multicolumn{2}{c|}{Model}')
for i in range(len(metrics)):
    f.write('&')
    f.write(metrics[i])
f.write('&Pattern Ratio')
f.write('\\\\\n')

f.write('\\midrule' + '\n')
f.write('\\multirow{%s}{*}{%s}'%(2, model_names[0]))

for j in range(len(typees[model_names[0]])):
    if j != 0:
        f.write('~')
    f.write('&%s'%typees[model_names[0]][j])
    cur_data = all_data[model_names[0]][j]
    f.write('&' + '&'.join(['%.2f'%x for x in cur_data]) + '\\\\\n')

f.write('\\midrule' + '\n')
f.write('\\multirow{%s}{*}{%s}'%(3, model_names[1]))

for j in range(len(typees[model_names[1]])):
    if j != 0:
        f.write('~')
    f.write('&%s'%typees[model_names[1]][j])
    cur_data = all_data[model_names[1]][j]
    f.write('&' + '&'.join(['%.2f'%x for x in cur_data]) + '\\\\\n')

f.write('\\midrule' + '\n')
f.write('\\multirow{%s}{*}{%s}'%(4, model_names[2]))

for j in range(len(typees[model_names[2]])):
    if j != 0:
        f.write('~')
    f.write('&%s'%typees[model_names[2]][j])
    cur_data = all_data[model_names[2]][j]
    f.write('&' + '&'.join(['%.2f'%x for x in cur_data]) + '\\\\\n')

f.write('\\midrule' + '\n')
f.write('\\multirow{%s}{*}{%s}'%(3, model_names[3]))

for j in range(len(typees[model_names[3]])):
    if j != 0:
        f.write('~')
    f.write('&%s'%typees[model_names[3]][j])
    cur_data = all_data[model_names[3]][j]
    f.write('&' + '&'.join(['%.2f'%x for x in cur_data]) + '\\\\\n')

f.write('\\bottomrule' + '\n')
f.write('\\end{tabular}' + '\n')

f.write('\\end{table}' + '\n')
f.close()