#!/bin/bash

i="1"

while [ $i -lt 10 ]
do
	cat logs/log$[i].txt | sed '$!d'
	i=$[$i+1]
done
