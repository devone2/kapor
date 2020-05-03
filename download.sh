#!/bin/bash

n=${2:-100}
mkdir -p unlabeled
echo "Number to download: $n"
for ((i=0;i<$n;i++))
do
         echo "Downloading image: $i"
         wget "https://www.katasterportal.sk/kapor/Captcha.jpg" -O unlabeled/cap_$i.jpg
done;
