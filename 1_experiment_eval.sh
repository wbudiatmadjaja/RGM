pwd
echo '----------start-----------'
echo 'ModelNet40_eval:'
echo 'Seen Clean input:'
python eval.py --cfg experiments/test_RGM_Seen_Clean_modelnet40_transformer.yaml
echo 'Seen Noisy input:'
python eval.py --cfg experiments/test_RGM_Seen_Jitter_modelnet40_transformer.yaml
echo 'Seen Partial input:'
python eval.py --cfg experiments/test_RGM_Seen_Crop_modelnet40_transformer.yaml