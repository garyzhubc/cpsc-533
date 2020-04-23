GPU=$1
for NUM_CAPS in 3 # 10 20
do
	for CAPS_CH in 8 # 16 64 128 160 200
	do
		for MASK in True #False # True
		do
			NUM_CAPS=${NUM_CAPS} CAPS_CH=${CAPS_CH} MASK=${MASK} CUDA_VISIBLE_DEVICES=$GPU python configs/train_encodeDecode.py
			# NUM_CAPS=${NUM_CAPS} CAPS_CH=${CAPS_CH} MASK=${MASK} CUDA_VISIBLE_DEVICES=$GPU python configs/train_encodeDecode_pose.py
			# CUDA_VISIBLE_DEVICES=$GPU python configs/train_encodeDecode_pose.py
		done
	done
done
