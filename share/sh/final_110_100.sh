#!/usr/bin/env zsh

config_path='share/experiments/cifar100-resnet110-final#.yml'
log_path='logs/exps/final_110_100.log'

if [ -f $log_path ]; then
  echo "Created file will be cleared."
  echo -n "" > $log_path
fi

nohup python main.py -y $config_path -c > $log_path 2>&1 &
