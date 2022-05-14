# different loss weight lambda1 and whether to use regularizier
for lambda1 in 0 1
do
    for lambda2 in 0 
    do
        for net in 'edsr' 'rdn'
        do
            # for spaloss in 'ssim' 'l2' 'perceptual' 'mix'
            for spaloss in 'l1' 'ssim' 
            do 
                for reg in 1
                do 
                    for aug in 0 1
                    do
                        echo 'lambda1 = ' $lambda1 'lambda2 = ' $lambda2 'net = ' $net 'spaloss = ' $spaloss 'reg = ' $reg  'aug = ' $aug
                        python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                    --batch-size 64 \
                                    --lr 1.6e-3 \
                                    --num-epochs 50 \
                                    --num-save 10 \
                                    --sample-interval 500 \
                                    --outputs-dir att_outputs/10000_wrf \
                                    --im_size 128 \
                                    --spa_loss $spaloss \
                                    --fre_loss focal_fft \
                                    --att_net $net \
                                    --train-length 10000 \
                                    --valid-length 500 \
                                    --lambda1 $lambda1 \
                                    --lambda2 $lambda2 \
                                    --reg $reg \
                                    --aug $aug
                    done
                done
            done
        done
    done
done