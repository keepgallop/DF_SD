# different loss weight lambda1 and whether to use regularizier
for lambda1 in 1
do
    for lambda2 in 5
    do
        for net in 'rdn'
        do
            # for spaloss in 'ssim' 'l2' 'perceptual' 'mix'
            for spaloss in 'mix'
            do 
                for reg in 1
                do 
                    for aug in 1
                    do
                        echo 'lambda1 = ' $lambda1 'lambda2 = ' $lambda2 'net = ' $net 'spaloss = ' $spaloss 'reg = ' $reg  'aug = ' $aug
                        python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                    --batch-size 64 \
                                    --lr 1.6e-3 \
                                    --num-epochs 50 \
                                    --num-save 10 \
                                    --sample-interval 500 \
                                    --outputs-dir att_outputs/10000_128input \
                                    --im_size 128 \
                                    --spa_loss $spaloss \
                                    --fre_loss focal_fft \
                                    --att_net $net \
                                    --train-length 10000 \
                                    --valid-length 1000 \
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