#!/bin/bash
# This is a script for getting raw data for training.

echo 'getting raw data began!'

for i in $(seq 1 0.05 2.01);
do

  new="aw=$i"
  sed -i '' '12s/.*/'$new'/' read-raw-data.py
  /Users/xuhuizhou/opt/anaconda3/bin/python /Users/xuhuizhou/working/periodicHill-auto/read-raw-data.py

done

echo 'getting raw data finished!'








