
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../

caffe/build/tools/caffe train -solver {{solver_file}} -gpu $1 -weights {{pre-trained}}
