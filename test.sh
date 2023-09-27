python3 \
  main.py \
  --mode test \
  --attr_path ./data/test_attributes.txt \
  --latent_path ./data/test_latents.npy \
  --latent_type wp \
  --sample_dir ./results/ \
  --checkpoint_dir "./pretrained_models/" \
  --resume_iter 100000 \
  --num_workers 1 \
  --stylegan2_path ./pretrained_models/stylegan2-ffhq-config-f.pt \
