for data_dir in 'dataset/attack_stage1_samples/rdn-ssim-fft/rdn/after/' 'dataset/attack_stage1_samples/unet-ssim-fft/unet/after/' 'dataset/attack_stage1_samples/vae-ssim-fft/vae/after/'
do
for net in 'efficientnet' 'resnet' 'xception' 
do
for fake_class in 'all' 'progan' 
        do
            python3 train_detector_on_attack.py --data_dir $data_dir \
                                        --batch-size 250 \
                                        --lr 1.6e-3 \
                                        --num-epochs 50 \
                                        --num-save 25 \
                                        --outputs-dir det_training_log/train_on_attack\
                                        --im_size 128 \
                                        --det_net $net \
                                        --length 20000 \
                                        --aug 0 \
                                        --opt adam \
                                        --fake-class $fake_class \
                                        --feature-space 'rgb'
        done
    done
done
