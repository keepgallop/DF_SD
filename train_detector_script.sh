for feature_space in 'rgb' 'dct' 
do
    for aug in 0 1
    do
        for net in 'xception'  'resnet' 
        do
            echo 'net = ' $net 'aug = ' $aug
            python3 train_detector.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                        --batch-size 250 \
                                        --lr 1.6e-3 \
                                        --num-epochs 50 \
                                        --num-save 2 \
                                        --outputs-dir det_outputs/all20000 \
                                        --im_size 128 \
                                        --det_net $net \
                                        --train-length 20000 \
                                        --valid-length 1000 \
                                        --aug $aug \
                                        --opt adam \
                                        --feature-space $feature_space \
                                        --fake-class 'progan' 'mmdgan' 'sngan' 'crgan'
        done
    done
done
