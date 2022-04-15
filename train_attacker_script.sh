# different loss weight lambda1 and whether to use regularizier
for lambda1 in 1
do
    for lambda2 in 1
    do
        for net in 'rdn'
        do
            # for spaloss in 'ssim' 'l2' 'perceptual' 'mix'
            for spaloss in 'mix'
            do 
                for reg in 1
                do 
                    for aug in 0
                    do
                        echo 'lambda1 = ' $lambda1 'lambda2 = ' $lambda2 'net = ' $net 'spaloss = ' $spaloss 'reg = ' $reg  'aug = ' $aug
                        python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                    --batch-size 128 \
                                    --lr 1e-4 \
                                    --num-epochs 50 \
                                    --num-save 5 \
                                    --sample-interval 500 \
                                    --outputs-dir att_outputs/5000 \
                                    --im_size 128 \
                                    --spa_loss $spaloss \
                                    --fre_loss focal_fft \
                                    --att_net $net \
                                    --train-length 5000 \
                                    --valid-length 2000 \
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