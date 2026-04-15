docker run -d --name FGIC --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/xz/wjl/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/xz/wjl/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/xz/wjl/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/xz/wjl/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh test400