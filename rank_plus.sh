#!/usr/bin/env zsh

./run.sh vgg16 e1
wait

./run.sh resnet56 e2
wait

./run.sh resnet110 e3
wait

./run.sh mobilenetV2 e4
wait
