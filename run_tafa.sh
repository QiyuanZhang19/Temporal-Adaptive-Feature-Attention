gpu_id=0
version='version'
mkdir -p "./logs/TAFA-${version}"
for env in "PendulumAug-v0" # "Walker2d-v2" "Hopper-v2"  "HalfCheetah-v2" "Ant-v2"
do
  for seed in 0 1
    do
      python main_train_tafa.py --gpu_id ${gpu_id} --seed ${seed} --env ${env} >> "./logs/TAFA-${version}/${env}--s-${seed}"  2>&1 &
      gpu_id=$((($gpu_id+1) % 2))
    done
done







