import sys
lemmatization = {"added": "add", "fixed": "fix", "removed": "remove", "adding": "add", "fixing": "fix", "removing": "remove"}
path = str(sys.argv[1])
output_path = str(sys.argv[2])
data = [x.strip().split(',')[0].split() for x in open(path) if x.strip()]
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = data[i][j].lower()
        if data[i][j] in lemmatization:
            data[i][j] = lemmatization[data[i][j]]
file_name = path.split('/')[-1]
if not file_name.startswith('output_'):
    file_name = output_path + '/output_' + file_name
f = open(file_name, 'w')
for i, x in enumerate(data):
    if i > 0:
        f.write('\n')
    
    f.write(' '.join(x))
    # f.write(' '.join(x) + '\n')