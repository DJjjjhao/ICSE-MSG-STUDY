import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
def plot_ratios_bleu(data, name):
    bleus = [i / 10 for i in range(0, 10)]
    plt.figure()
    for i in range(len(model_names)):
        plt.plot(bleus, data[i], label=model_names[i])  
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.savefig('%s.png'%name)
if __name__ == '__main__':
    model_names = ['nmt', 'ptrgn', 'codisum', 'coregen', 'fira']
    output_names = {'nmt':'NMT', 'ptrgn':'PtrGNCMsg', 'codisum':'CODISUM', 'coregen':'CoreGen', 'fira':'FIRA'}

    total_ratios = []
    
    for i in range(len(model_names)):
        model_name = model_names[i]
        cur_ratios = eval(open('../Patterns/pattern_bleu_%s_train'%model_name).read())
        cur_ratios = [float(x) for x in cur_ratios]
        cur_ratios = [float(x) * 100 for x in cur_ratios]


        total_ratios.append(cur_ratios)
        bleus = [i * 10 for i in range(0, 10)]
    # mpl.style.use('seaborn')
    plt.figure()
    CB91_Amber = '#F5B14C'    
    colors = ['b', 'g', 'c', CB91_Amber, 'tab:purple']
    for i in range(len(model_names)):
        plt.plot(bleus, total_ratios[i], color=colors[i], linewidth=2, label=output_names[model_names[i]])  
    # plt.ylim(top=1)
    plt.xlabel('BLEU')
    plt.ylabel('Ratio of patterns (%)')
    # plt.title("")
    leg = plt.legend(fontsize=8,frameon=False, ncol =2)
    # prop=dict(weight='bold')
    # plt.savefig('pattern_bleu.pdf')
    plt.savefig('TablesAndFigures/rq1_figure4.png')
