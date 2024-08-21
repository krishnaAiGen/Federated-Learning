#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:32:10 2023

@author: krishna
"""

attack_dictionary = {}

from collections import Counter
for data in y_data:
    count = Counter(data)
    for key in count.keys():
        if key in attack_dictionary:
            attack_dictionary[key] = attack_dictionary[key] + count[key]
            
        else:
            attack_dictionary[key] = 0
            attack_dictionary[key] = attack_dictionary[key] + count[key]