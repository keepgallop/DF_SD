# different loss weight lambda1 and whether to use regularizier
for lambda1 in 0 0.1 0.5 1 2 5 10
do
    for lambda2 in 0 1
    do
        for net in 'rdn' 'vae'
        do
            for spaloss in 'ssim' 'l2' 'perceptual' 'mix'
            do 
                if [ $lambda2 -eq 0 ];then 
                    echo 'lambda1 = ' $lambda1 ', no reg'
                    python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                --batch-size 128 \
                                --lr 1e-3 \
                                --num-epochs 30 \
                                --num-save 5 \
                                --sample-interval 500 \
                                --outputs-dir att_outputs \
                                --im_size 128 \
                                --spa_loss $spaloss \
                                --fre_loss focal_fft \
                                --att_net $net \
                                --train-length 20000 \
                                --valid-length 2000 \
                                --lambda1 $lambda1 \
                                --lambda2 $lambda2 \
                                
                else 
                    echo 'lambda1 = ' $lambda1 ', with reg'
                    python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                --batch-size 128 \
                                --lr 1e-3 \
                                --num-epochs 30 \
                                --num-save 5 \
                                --sample-interval 500 \
                                --outputs-dir att_outputs \
                                --im_size 128 \
                                --spa_loss $spaloss \
                                --fre_loss focal_fft \
                                --att_net $net \
                                --reg \
                                --train-length 20000 \
                                --valid-length 2000 \
                                --lambda1 $lambda1 \
                                --lambda2 $lambda2 \

                fi
            done
        done
    done
done