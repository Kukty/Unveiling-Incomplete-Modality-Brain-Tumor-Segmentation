#Without pretrain , unimodel 
python main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir unimodel_flair --save_checkpoint --in_modalities Flair \
--batch_size 1 --distributed \
--checkpoint /root/SwimUNETR_miss/BraTS2018/runs/unimodel_flair/model_final.pt

#finetune best full modality model 
python main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir ft_full --save_checkpoint \
--in_modalities Flair T1 T1c T2 \
--batch_size 1 --distributed \
--checkpoint /root/SwimUNETR_miss/BraTS2021/runs/test/model_final.pt


# train by adding kd_loss 
python kl_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir kd_flair --save_checkpoint --in_modalities Flair \
--batch_size 1 --distributed \
--pretrain_teacher=/root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt

#train by adding mse between hidden states
python hs_dis_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir mse_flair --save_checkpoint --in_modalities Flair \
--batch_size 1 --distributed \
--pretrain_teacher=/root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt


# Pretrain to from flair to rec full modality 
python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_rec_flair --save_checkpoint --in_modalities Flair \
--batch_size 1 --distributed 

# use rec flair to full modality pretrain encoder 
python main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir unimodel_flair --save_checkpoint --in_modalities Flair \
--batch_size 1 --distributed \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pre_train_swinVit_rec_flair.pt

# Train only on T1 unimodel 
python main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir unimodel_T1 --save_checkpoint --in_modalities T1 \
--batch_size 1 

# Pretrain to from random singal modality to rec full modality 
python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_random_rec_singal_modality --save_checkpoint --in_modalities Flair \
--batch_size 2 \
--checkpoint /root/SwimUNETR_miss/BraTS2018/runs/pre_random_rec_singal_modality/model_final.pt

python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_rec_full_modality --save_checkpoint \
--in_modalities Flair T1 T1c T2 \
--batch_size 2

# Pretrain from random singal modality ro rec remains 3 modalities
python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_rec_3_modality_multistage --save_checkpoint \
--in_modalities Flair \
--batch_size 2 --distributed --max_epochs 800

#train by adding mse between and kl hidden states and pretraning T1 
python hs_dis_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir mse_kl_t1 --save_checkpoint --in_modalities T1 \
--batch_size 1 --distributed \
--pretrain_teacher=/root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pre_train_swinVit_multista_singal2three.pt \
--checkpoint /root/SwimUNETR_miss/BraTS2018/runs/mse_kl_t1/model_final.pt

# Pretrain from masked singal modality to rec all modalities
python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_masked_flair --save_checkpoint \
--in_modalities Flair \
--batch_size 2 --distributed --max_epochs 800

# train by adding kd_loss with masked AE (--usd_kl)
python kl_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir kd_flair_masked --save_checkpoint --in_modalities Flair \
--batch_size 1 \
--pretrain_teacher /root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pre_train_masked_flair.pt \
--checkpoint /root/SwimUNETR_miss/BraTS2018/runs/kd_flair_masked/model_final.pt

#(--add_holder)(XXXX)
python kl_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir holder_flair_masked --save_checkpoint --in_modalities Flair \
--batch_size 1 --use_holder --kd_weight 1.0 \
--pretrain_teacher /root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pre_train_masked_flair.pt \

# Pretrain from masked two modality to rec all modalities
python pre_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir pre_masked_flair_t1 --save_checkpoint \
--in_modalities Flair T1 \
--batch_size 2 --max_epochs 800

## train by adding kd_loss with masked AE (--usd_kl) _flair _ t1
python kl_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir kd_flair_t1_masked --save_checkpoint --in_modalities Flair T1 \
--batch_size 1 --use_kl \
--pretrain_teacher /root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pre_train_masked_flair_t1.pt

## train by adding kd_loss with masked AE (--usd_kl) _flair _ t1
python kl_main.py --json_list=/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json --data_dir=/root/autodl-tmp/BraTs2018/BraTs2018/ --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128 --spatial_dims=3 --use_checkpoint --feature_size=48 \
--logdir kd_flair_t1_t1c_masked --save_checkpoint --in_modalities Flair T1 T1c \
--batch_size 1 --use_kl \
--pretrain_teacher /root/SwimUNETR_miss/BraTS2018/runs/ft_full/model_final.pt \
--pretrain_vit /root/SwimUNETR_miss/BraTS2018/pretrained_models/pretrain_mask_flair_t1c_t1.pt