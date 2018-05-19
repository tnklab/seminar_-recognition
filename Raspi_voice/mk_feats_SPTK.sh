#!/bin/bash




datadir=$1
outdir=$2

mkdir $outdir

for wav_file in `find ${datadir} | grep wav`
do
    wav_name=`echo ${wav_file}|python -c $'import sys\nfor line in sys.stdin:\tprint(line.strip().split("/")[-1].split(".")[0])'`
    echo "Working with ${wav_name}"
    # echo $wav_name
    cat $wav_file | x2x +sf | frame -l 640 -p 160 | mfcc -l 640 -f 16 -m 13 -n 20 -a 0.97 > $outdir/data.mfcc
    python print_mfcc.py $outdir/data.mfcc 13 | perl normalize_SPTK_modified_print.pl | sed "s/\t/ /g"> $outdir/mfcc.csv
    rm $outdir/data.mfcc
    cat $outdir/mfcc.csv > $outdir/${wav_name}.mfcc
    rm $outdir/mfcc.csv
done
echo "Extract Done"
