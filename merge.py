# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:10:29 2018

@author: vishnu
"""

#import csv
    


fout=open("final.csv","a")
for num in range(1,11):
    for line in open("D"+str(num)+".csv"):
         fout.write(line)    
fout.close()

