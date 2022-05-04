for feature_space in 'rgb' #'dct' 
do
    for aug in 0 1
    do
        for net in 'resnet' # 'resnet' 
        do
            echo 'net = ' $net 'aug = ' $aug
            python3 train_detector_on_attack.py --data-csv-file dataset/celeba-128/celeba_128.csv \
                                        --batch-size 250 \
                                        --lr 1.6e-3 \
                                        --num-epochs 100 \
                                        --num-save 10 \
                                        --outputs-dir det_outputs/train_on_attack\
                                        --im_size 128 \
                                        --det_net $net \
                                        --train-length 0 \
                                        --valid-length 0 \
                                        --aug $aug \
                                        --opt adam \
                                        --fake-class 'all' \
                                        --feature-space $feature_space
        done
    done
done
