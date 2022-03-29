# different loss weight lambda1 and whether to use regularizier
for lambda1 in 0 0.1 0.5 1 2 5 10
do
    for lambda2 in 0 $lambda1
    do
        if [ $lambda2 -eq 0 ];then 
            echo 'lambda1 = ' $lambda1 ', no reg'
            python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                        --batch-size 128 \
                        --lr 1.6e-3 \
                        --num-epochs 30 \
                        --num-save 5 \
                        --sample-interval 50 \
                        --augment \
                        --outputs-dir att_outputs \
                        --im_size 128 \
                        --spa_loss perceptual \
                        --fre_loss focal_fft \
                        --att_net unet \
                        --train-length 20000 \
                        --valid-length 2000 \
                        --lambda1 $lambda1 \
                        --lambda2 $lambda2 \
                        
        else 
            echo 'lambda1 = ' $lambda1 ', with reg'
            python3 train_attacker.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                        --batch-size 128 \
                        --lr 1.6e-3 \
                        --num-epochs 30 \
                        --num-save 5 \
                        --sample-interval 50 \
                        --augment \
                        --outputs-dir att_outputs \
                        --im_size 128 \
                        --spa_loss perceptual \
                        --fre_loss focal_fft \
                        --att_net unet \
                        --reg \
                        --train-length 20000 \
                        --valid-length 2000 \
                        --lambda1 $lambda1 \
                        --lambda2 $lambda2 \
                        
        fi
    done
done