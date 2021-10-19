#!/bin/bash
# This is a script for getting raw data for training.

echo 'do general validation with full size!'

for i in $(seq 2.6 0.1 3.51);
do

  new="aw=$i"
  sed -i '' '15s/.*/'$new'/' general-validation.py
  /Users/xuhuizhou/opt/anaconda3/bin/python /Users/xuhuizhou/working/periodicHill-auto/general-validation.py

done

# for i in $(seq 2.05 0.05 2.55);
# do

#   new="aw=$i"
#   sed -i '' '15s/.*/'$new'/' general-validation.py
#   /Users/xuhuizhou/opt/anaconda3/bin/python /Users/xuhuizhou/working/periodicHill-auto/general-validation.py

# for i in $(seq 1.075 0.2 1.9);
# do

#   new="aw=$i"
#   sed -i '' '15s/.*/'$new'/' general-validation.py
#   /Users/xuhuizhou/opt/anaconda3/bin/python /Users/xuhuizhou/working/periodicHill-auto/general-validation.py

# done
# for i in $(seq 3.0 0.1 3.0);
# do

#   new="aw=$i"
#   sed -i '' '15s/.*/'$new'/' general-validation.py
#   /Users/xuhuizhou/opt/anaconda3/bin/python /Users/xuhuizhou/working/periodicHill-auto/general-validation.py

# done

echo 'general validation finished!'