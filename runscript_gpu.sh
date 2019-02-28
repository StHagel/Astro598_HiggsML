#!/bin/bash

echo Running with tags IGNORE_JET_DATA and IGNORE_MASS_DATA, writing output to logs/log1.txt

python3 src/main_gpu.py IGNORE_JET_DATA IGNORE_MASS_DATA > logs/log1.txt

echo Running with tags IGNORE_JET_DATA and REMOVE_HIGGS_NAN, writing output to logs/log2.txt

python3 src/main_gpu.py IGNORE_JET_DATA REMOVE_HIGGS_NAN > logs/log2.txt

echo Running with tags IGNORE_JET_DATA and SIMPLE_IMPUTE, writing output to logs/log3.txt

python3 src/main_gpu.py IGNORE_JET_DATA SIMPLE_IMPUTE > logs/log3.txt


echo Running with tags IGNORE_MULTIJET_DATA and IGNORE_MASS_DATA, writing output to logs/log4.txt

python3 src/main_gpu.py IGNORE_MULTIJET_DATA IGNORE_MASS_DATA > logs/log4.txt

echo Running with tags IGNORE_MULTIJET_DATA and REMOVE_HIGGS_NAN, writing output to logs/log5.txt

python3 src/main_gpu.py IGNORE_MULTIJET_DATA REMOVE_HIGGS_NAN > logs/log5.txt

echo Running with tags IGNORE_MULTIJET_DATA and SIMPLE_IMPUTE, writing output to logs/log6.txt

python3 src/main_gpu.py IGNORE_MULTIJET_DATA SIMPLE_IMPUTE > logs/log6.txt


echo Running with tags IGNORE_MASS_DATA, writing output to logs/log7.txt

python3 src/main_gpu.py IGNORE_MASS_DATA > logs/log7.txt

echo Running with tags REMOVE_HIGGS_NAN, writing output to logs/log8.txt

python3 src/main_gpu.py REMOVE_HIGGS_NAN > logs/log8.txt

echo Running with tags SIMPLE_IMPUTE, writing output to logs/log9.txt

python3 src/main_gpu.py SIMPLE_IMPUTE SAVE_MODEL > logs/log9.txt

