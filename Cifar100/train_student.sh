#! /bin/bash

#!  /opt/conda/bin/python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
	# --distill ickd \
	# --model_s resnet8x4 \
	# -a 0 -b 4 --trial 1
	#-a 2 -b 4 --trial 1
#! /bin/bash

/opt/conda/bin/python train_student.py \
	--path_t ./save/teacher_resnet32x4_best.pth \
	--distill dckd \
	--model_s resnet8x4 \
	--model_t resnet32x4 \
	-a 0.5 -b 0.5 --trial 1
