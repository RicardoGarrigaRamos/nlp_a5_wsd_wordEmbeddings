"""
scorer.py
Ricardo Garriga-Ramos
CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - feb 12 Spring 2024


compare my-line-answers.txt to line-key.txt
reports the overall accuracy of your tagging, and provide a confusion matrix

python3 scorer.py my-line-answers.txt line-key.txt

"""

import sys
import re

# open line key and my line answers
out = open(sys.argv[1], 'r')
key = open(sys.argv[2], 'r')

# true positive
tp = 0
# true negitive
tn = 0
# false positive
fp = 0
# false negitive
fn = 0
count = 0


while True:
    # gets current instence from line key that matches my line answers
    cur_out = out.readline()
    cur_key = key.readline()

    if not cur_key or not cur_out:
        break
    count += 1
    phone_out = re.search("phone", cur_out)
    phone_key = re.search("phone", cur_key)
    product_out = re.search("product", cur_out)
    product_key = re.search("product", cur_key)


    # deside where it goes on the econfusion matrix
    if phone_key and phone_out:
        tp += 1
    elif product_key and product_out:
        tn += 1
    elif product_key and phone_out:
        fp += 1
    elif phone_key and product_out:
        fn += 1
    else:
        print("error")
        break
    
# print confusion matrix
print("Confusion matrix were phone is positive and product is Negitive")
print(f"True positive = {tp}\tTrue Negitive = {tn}\nFalse Positive = {fp}\tFalse Negitive = {fn}")
print(f"Accuracy = {(tp+tn)/(tp+tn+fp+fn)}")
    
 

out.close()
key.close()