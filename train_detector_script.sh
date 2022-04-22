for net in 'xception' 'resnet' 'efficientnet'
do
    for aug in 1 0
    do
        for feature_space in 'dct' 'rgb' 
        do
            echo 'net = ' $net 'aug = ' $aug
            python3 train_detector.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                        --batch-size 128 \
                                        --lr 1.6e-3 \
                                    --num-epochs 50 \
                                    --num-save 10 \
                                    --outputs-dir det_outputs/20000 \
                                    --im_size 128 \
                                    --det_net $net \
                                    --train-length 20000 \
                                    --valid-length 2000 \
                                    --aug $aug \
                                    --opt sgd \
                                    --fake-class 'progan' \
                                    --feature-space $feature_space
        done
    done
done