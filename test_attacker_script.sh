
for d in './det_outputs/all20000/x128-xception-aug-0-proganmmdgansngancrgan-rgb/best.pth' \
'./det_outputs/full_dataset/x128-resnet-aug-0-all-rgb/best.pth' \
'./det_outputs/full_dataset/x128-efficientnet-aug-0-all-rgb/epoch_49.pth' \
'./det_outputs/all10000/x128-resnet-aug-0-all-rgb/best.pth' \
'./det_outputs/train_on_attack/x128-resnet-aug-0-all-rgb/best.pth' \
'./det_outputs/train_on_attack/x128-xception-aug-0-stgan-rgb/best.pth'
do
    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/vae-ssim/' \
                            --attacker-ckpt './att_outputs/x128-vae-ssim-focal_fft-lambda1-0.0-reg-0-aug-False/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'xception' \
                            --att_net 'vae' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'

    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/vae-ssim-fft/' \
                            --attacker-ckpt './att_outputs/x128-vae-ssim-focal_fft-lambda1-2.0-reg-1-aug-False/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'resnet' \
                            --att_net 'vae' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'

    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/unet-ssim/' \
                            --attacker-ckpt './att_outputs/x128-unet-perceptual-focal_fft-lambda1-0.0-reg-0-noaug/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'efficientnet' \
                            --att_net 'unet' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'

    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/unet-ssim-fft/' \
                            --attacker-ckpt './att_outputs/x128-unet-perceptual-focal_fft-lambda1-2.0-reg-1-noaug/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'resnet' \
                            --att_net 'unet' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'


    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/rdn-ssim/' \
                            --attacker-ckpt './att_outputs/x128-rdn-ssim-focal_fft-lambda1-0.0-reg-0-aug-False/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'resnet' \
                            --att_net 'rdn' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'

    python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                            --outputs-dir './test_results/all_training_samples/rdn-ssim-fft/' \
                            --attacker-ckpt './att_outputs/x128-rdn-ssim-focal_fft-lambda1-2.0-reg-1-aug-False/epoch_29.pth' \
                            --detector-ckpt $d \
                            --det_net  'xception' \
                            --att_net 'rdn' \
                            --feature-space 'rgb' \
                            --new-attack 'false' \
                            --allocation 'attack-on-train'