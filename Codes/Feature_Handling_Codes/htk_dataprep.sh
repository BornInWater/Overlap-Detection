HTKDIR="/home/neerajs/htk/HTKTools/"

$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S /home/neerajs/work/blurp_universe/splitaa &
$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S /home/neerajs/work/blurp_universe/splitab &
$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S /home/neerajs/work/blurp_universe/splitac &
$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S /home/neerajs/work/blurp_universe/splitad &
$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S /home/neerajs/work/blurp_universe/splitae &





#$HTKDIR/HList -h -z feat/p306_359.htk

#if [ -d hmm0 ]
#then 
 #  rm -f hmm0/*
#else
#    mkdir hmm0
#fi

#$HTKDIR/HCompV -C config -f 0.01 -m -S f.scp -M hmm0 proto

#/home/neeks/work/ami/feats/fbank/train/
