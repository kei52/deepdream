#!usr/bin/env 8 python
# -*- coding: utf-8 -*-

#['input']
#len11
name1 = ['conv2d0','maxpool0','localresponsenorm0','conv2d1','conv2d2','localresponsenorm1','maxpool1',
         'mixed3a','mixed3b','maxpool4','mixed4a']

#name1_1 = ['head0','nn0','softmax0','output']
#len3,
name2 = ['mixed4b','mixed4c','mixed4d']

#name2_1 = ['head1','nn1','softmax1','output1']
#len7
name3 = ['maxpool10','mixed5a','mixed5b','avgpool0','softmax2','output2']

"""
name4 = ['','','','','','','','','','','','','','',
         '','','','','','','','','','','','','','',
         '','','','','','','','','','','','','','']
"""
LAYER_NAME = {}
for (i,x) in enumerate(name1):
    LAYER_NAME.update({i:x})
    #print LAYER_NAME
