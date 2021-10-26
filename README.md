# Fast Virtual Simulator for Safe Quadruped Locomotion in an Unknown Complex Environment

Explanation for contents **"env/envs/"**.

**1. anymal_flat_model**
  - Testing codes quadruped robot command tracking controller in flat terrain

**2. lidar_model**
  - Training and testing codes for *FAVISQ* in parameterized environments
  - Testing is done in environments with cylinders and boxes

**3. lidar_model_baseline**
  - Tesing codes for baseline(CWM) in environments with cylinders and boxes

**4. lidar_model_baseline_potential**
  - Tesing codes for baseline(Potential function) in environments with cylinders and boxes
  - Simple PD controller was used to generate commands for tracking the generated gradient direction
  - The corresponding baseline has been excluded from the paper because of unstable behavior

**5. lidar_model_baseline_sim_wo_map**
  - Tesing codes for baseline(CWOM) in environments with cylinders and boxes

**6. lidar_model_test**
  - Testing codes for *FAVISQ* in complex environments

**7. lidar_model_test_baseline**
  - Testing codes for baseline(CWM) in complex environments

**8. lidar_model_test_baseline_sim_wo_map**
  - Testing codes for baseline(CWOM) in complex environments
