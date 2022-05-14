python3 train_attacker_stage2.py --data-path 'test_results/10000_wrf/rdn' \
                                    --batch-size 64 \
                                    --lr 1.6e-3 \
                                    --num-epochs 50 \
                                    --num-save 10 \
                                    --sample-interval 500 \
                                    --outputs-dir 'att_stage2_wrf_outputs' \
                                    --im_size 128 \
                                    --spa_loss 'mix' \
                                    --fre_loss 'focal_fft' \
                                    --att_net 'stage2_rdn' \
                                    --length 0 \
                                    --lambda1 1 \
                                    --lambda2 5 \
                                    --reg 0 \
                                    --aug 'false'