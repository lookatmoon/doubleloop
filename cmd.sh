
work_dir=./G151

npos_eval=992
nneg_eval=240332
# time python dataset_utility.py make_eval $npos_eval $nneg_eval $work_dir/eval.txt

npos_seed=10
nneg_seed=10
time python dataset_utility.py make_seed $npos_seed $nneg_seed $work_dir/seed.txt


