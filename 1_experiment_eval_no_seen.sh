pwd
echo '----------start-----------'
echo 'ModelNet40_eval:'
echo 'Unseen Crop Transformer:'
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_transformer.yaml
echo 'Unseen Crop NoAttention:'
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAttention.yaml
echo 'Unseen Crop NoAttention NN:'
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAttention_nn.yaml
echo 'Unseen Crop NoAIs:'
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAIS.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAIS.yaml
