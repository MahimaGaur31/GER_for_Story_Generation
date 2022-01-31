#!/bin/bash

#Change to your netid if your scratch directory does not exist then create a directory first
#cd /scratch/{netid}
cd /scratch/vk2161/entropy/

#Clone the repository
git clone https://github.com/rycolab/entropyRegularization.git

#change the directory to the folder
cd entropyRegularization

#Installs fairseq and its components
pip install --editable .

#change directory to Hierarichal tory example
cd examples/stories

#Download the dataset
curl https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar xvzf -

echo "Downloading the dataset completed"
echo "Changing the directory to writingPrompts"

#cd to the dataset to run preprocessing
cd writingPrompts

echo "Changed the directory to writingPrompts"
echo "Creating a file script.py"

#create a python script file
touch script.py

#Python code to trim the dataset to the first 1000 words of each story and just take 0.04% of the dataset
echo -e "import math

data = [\"train\", \"test\", \"valid\"]
for name in data:
    with open(name + \".wp_target\") as f:
        stories = f.readlines()
    stories = [\" \".join(i.split()[0:1000]) for i in stories]
    print(name,\":\",len(stories))
    stories_size = int(math.floor(len(stories) * 0.04))
    stories = stories[:stories_size]
    print(name,\":\",len(stories))
    with open(name + \".wp_target\", \"w\") as o:
        for line in stories:
            o.write(line.strip() + \"\\\n\")

source = [\"train\", \"test\", \"valid\"]
for name in source:
    with open(name + \".wp_source\") as f:
        captions = f.readlines()
    caption_size = int(math.floor(len(captions) * 0.04))
    print(name,\":\",len(captions))
    captions = captions[:caption_size]
    print(name,\":\",len(captions))
    with open(name + \".wp_source\", \"w\") as o:
        for line in captions:
            o.write(line.strip() + \"\\\n\")" > script.py

echo "finished writing to script.py"
echo "Running the script.py"

#Run the script
python script.py

#export the dataset folder as a variable
export TEXT=examples/stories/writingPrompts

echo "changing 3 directories back"

#change directory back to entropyRegularization
cd ../../../


echo "starting preprocessing for the dataset"

#fairseq preprocessing the dataset, this will take sometime
fairseq-preprocess --source-lang wp_source --target-lang wp_target --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/writingPrompts --padding-factor 1 --thresholdtgt 10 --thresholdsrc 10


echo "preproessing done"
echo "starting the training"

#Now start the training
fairseq-train data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 --clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau --decoder-attention True --encoder-attention False --weight-decay .0000001 --source-lang wp_source --target-lang wp_target --gated-attention True --self-attention True --project-input True --pretrained False --criterion jensen_cross_entropy  --alpha 0.5 --beta 0.7 --use-uniform

echo "training done"

