#!/bin/bash
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

# global set

lmdb_path=dataset/train_lmdb/

arr=(${DIR//// })
num_arr=${#arr[@]}

model_name=${arr[$num_arr-2]}_${arr[$num_arr-1]}
echo model_name: $model_name


# exp set

read -p "Choose sets[default:1]:" word
if  [ ! -n "$word" ] ;then
    set_no=1
else
    set_no=$word
fi
exp_dir=experiments/$model_name/set$(printf "%02d" $set_no)
echo set_no: $set_no
exp_path=../../../$exp_dir

solver_path=$exp_dir/solver.prototxt
solver_path_hnm=$exp_dir/solver_hnm.prototxt
model_path=$exp_dir/train_val.prototxt
model_path_hnm=$exp_dir/train_val_hnm.prototxt
Snapshots_path=$exp_dir/Snapshots/set$(printf "%02d" $set_no)
Snapshots_path_hnm=$exp_dir/Snapshots/set$(printf "%02d" $set_no)_hnm

echo solver_path: $solver_path
echo model_path: $model_path
echo Snapshots_path: $Snapshots_path

echo solver_path_hnm: $solver_path_hnm
echo model_path_hnm: $model_path_hnm
echo Snapshots_path_hnm: $Snapshots_path_hnm

# generate lmdb? 
flag_data=no

# lmdb path
read -e -p "Input data path:[default:$lmdb_path]:" data_path
if  [ ! -n "$data_path" ] ;then
    data_path=../../../$lmdb_path
    train_list=$lmdb_path/set$(printf "%02d" $set_no)_train
    test_list=$lmdb_path/set$(printf "%02d" $set_no)_test
    train_list_hnm=$lmdb_path/set$(printf "%02d" $set_no)_train_hnm
    test_list_hnm=$lmdb_path/set$(printf "%02d" $set_no)_test_hnm
else
    lmdb_path=$data_path
    train_list=$data_path/set$(printf "%02d" $set_no)_train
    test_list=$data_path/set$(printf "%02d" $set_no)_test
    train_list_hnm=$data_path/set$(printf "%02d" $set_no)_train_hnm
    test_list_hnm=$data_path/set$(printf "%02d" $set_no)_test_hnm
fi

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
    echo makepath: $data_path
fi

cd ../../..
read -e -p "Input pretrained caffemodel path:" pretrained_path
cd $DIR
echo "pre-trained path:$pretrained_path"


if [ -f "$exp_path" ]; then
    rm "$exp_path"
fi

if [ ! -d "$exp_path" ]; then
    mkdir -p "$exp_path"
fi
if [ -d "$Snapshots_path" ]; then
    rm -r "$Snapshots_path"
fi
mkdir -p "$exp_path/Snapshots"
cp solver.prototxt $exp_path 
cp train_val.prototxt $exp_path
cp train_model.sh $exp_path
cp deploy.prototxt $exp_path

cp solver_hnm.prototxt $exp_path 
cp train_val_hnm.prototxt $exp_path
cp train_model_hnm.sh $exp_path


sed -i "s#{{net_path}}#$model_path#g" $exp_path/solver.prototxt 
sed -i "s#{{snapshots_prefix}}#$Snapshots_path#g" $exp_path/solver.prototxt 
sed -i "s#{{train_source}}#$train_list#g" $exp_path/train_val.prototxt 
sed -i "s#{{test_source}}#$test_list#g" $exp_path/train_val.prototxt
sed -i "s#{{solver_file}}#$solver_path#g" $exp_path/train_model.sh 

sed -i "s#{{net_path}}#$model_path_hnm#g" $exp_path/solver_hnm.prototxt 
sed -i "s#{{snapshots_prefix}}#$Snapshots_path_hnm#g" $exp_path/solver_hnm.prototxt 
sed -i "s#{{train_source}}#$train_list_hnm#g" $exp_path/train_val_hnm.prototxt 
sed -i "s#{{test_source}}#$test_list_hnm#g" $exp_path/train_val_hnm.prototxt
sed -i "s#{{solver_file}}#$solver_path_hnm#g" $exp_path/train_model_hnm.sh 
sed -i "s#{{pre-trained}}#$pretrained_path#g" $exp_path/train_model_hnm.sh 
