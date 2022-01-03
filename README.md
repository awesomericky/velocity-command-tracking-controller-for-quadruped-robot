# Quadruped command tracking controller (flat terrain)

### Prepare
1. Install RAISIM [link](https://raisim.com/sections/Installation.html)
2. Move to YOUR_PATH/raisimLib/raisimGymTorch/.
3. Remove everything in YOUR_PATH/raisimLib/raisimGymTorch/.
4. Clone the repository in YOUR_PATH/raisimLib/raisimGymTorch/.
5. Build environment
6. Train / Evaluate

cf)
- Trained model will be saved in YOUR_PATH/raisimLib/raisimGymTorch/data/command_tracking_flat/.
- Command tracking plot will be saved in YOUR_PATH/raisimLib/raisimGymTorch/command_tracking_plot/command_tracking_flat/.

### Build environment
```
python setup.py develop
```

### Train
```
python raisimGymTorch/env/envs/command_tracking_flat/runner.py
```

### Test
```
python raisimGymTorch/env/envs/command_tracking_flat/tester.py -w /home/awesomericky/raisim/raisimLib/raisimGymTorch/data/command_tracking_flat/2021-07-15-21-15-21/full_16200.pt
```

### Trained weights
Available in data.zip
