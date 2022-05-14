python3 test_attacker_stage2.py --data-path 'test_results/10000_wrf/rdn' \
                            --outputs-dir './test_stage2_results/test2-att_stage2_wrf_outputs' \
                            --data_length 1000 \
                            --attacker-ckpt 'att_stage2_wrf_outputs/x128-stage2_rdn-l1-focal_fft-lambda1-1.0-lambda2-0.0-reg-0-aug-0/best.pth' \
                            --detector-ckpt './det_outputs/train_on_attack_progan/x128-xception-aug-0-stgan-rgb/best.pth' \
                            --det_net  'xception' \
                            --att_net 'stage2_rdn' \
                            --feature-space 'rgb' \
                            --new-attack 'true' 