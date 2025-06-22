#!/bin/bash
cd /content/ICKD-DCKD/Cifar100

python train_student.py \
    --path_t ./save/teacher_resnet32x4_best.pth \
    --distill dckd \
    --model_s resnet8x4 \
    --model_t resnet32x4 \
    -a 0.5 -b 0.5 --trial 1
