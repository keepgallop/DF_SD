python3 test_attacker.py --data-csv-file './dataset/celeba-128/celeba_128.csv' \
                        --outputs-dir './test_results/' \
                        --attacker-ckpt './att_outputs/full_dataset/select-x128-rdn-mix-focal_fft-lambda1-1.0-lambda2-1.0-reg-1-aug-0/epoch_49.pth' \
                        --detector-ckpt './det_outputs/20000/x128-xception-aug-0-progan-dct/epoch_49.pth' \
                        --det_net  'xception' \
                        --att_net 'rdn' \
                        --feature-space 'dct' \
                        --fake-class 'progan' \
                        --new-attack 'true' 