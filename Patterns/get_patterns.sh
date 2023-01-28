# !/bin/sh

mkdir results

python get_text_spmf.py $1
java -jar spmf.jar run MaxSP  $1_spmf.text results/$1_maximal 2%
python analyze_results.py results/$1_maximal test  0
rm $1_spmf.text results/$1_maximal