#!/usr/bin/env zsh
# conda init zsh
# conda activate py36

if [ -f $1.log ]; then
  echo "Created file will be cleared."
  echo -n "" > $1.log
fi
nohup python main.py -y $2 > $1.log 2>&1 &
