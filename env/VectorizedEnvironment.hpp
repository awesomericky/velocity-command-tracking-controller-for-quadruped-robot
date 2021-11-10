//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"
#include <time.h>

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir), cfgString_(cfg) {
    Yaml::Parse(cfg_, cfg);
	raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {

    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    double min_obstacle_grid_size = cfg_["obstacle_grid_size"]["min"].template As<double>();
    double max_obstacle_grid_size = cfg_["obstacle_grid_size"]["max"].template As<double>();
    double min_obstacle_dr = cfg_["obstacle_dr"]["min"].template As<double>();
    double max_obstacle_dr = cfg_["obstacle_dr"]["max"].template As<double>();

    /// Set seed and obstacle grid size for generating random environment
    bool evaluate = cfg_["evaluate"].template As<bool>();
    int generator_seed;   /// Change the main seed to change the generated environment
    if (evaluate)
        generator_seed = cfg_["seed"]["evaluate"].template As<int>();
    else
        generator_seed = cfg_["seed"]["train"].template As<int>();
    std::cout << "Evaluate: " << evaluate << "\n";
    std::cout << "Seed: " << generator_seed << "\n";
    std::default_random_engine generator(generator_seed);
    std::uniform_int_distribution<> seed_uniform(0, 1000000);
    std::uniform_int_distribution<> env_type_uniform(1, 3);
    std::uniform_real_distribution<> obstacle_grid_size_uniform(min_obstacle_grid_size, max_obstacle_grid_size);
    std::uniform_real_distribution<> obstacle_dr_uniform(min_obstacle_dr, max_obstacle_dr);
    std::vector<int> seed_seq = {};
    std::vector<int> env_type = {};
    std::vector<double> obstacle_grid_size_seq = {};
    std::vector<double> obstacle_dr_seq = {};
    int cfg_determine_env = cfg_["determine_env"].template As<int>();
    for (int i=0; i<num_envs_; i++) {
        seed_seq.push_back(seed_uniform(generator));
        if (cfg_determine_env == 0) {
            if (i <= int(num_envs_ / 3))
                env_type.push_back(1);
            else if (i <= int(2 * num_envs_ / 3))
                env_type.push_back(2);
            else
                env_type.push_back(3);
        } else {
            env_type.push_back(cfg_determine_env)
        }
        obstacle_grid_size_seq.push_back(obstacle_grid_size_uniform(generator));
        obstacle_dr_seq.push_back(obstacle_dr_uniform(generator));
    }

    float n_type_1 = 0;
    float n_type_2 = 0;
    float n_type_3 = 0;
    for (int i=0; i<num_envs_; i++) {
        if (env_type[i] == 1)
            n_type_1 += 1.;
        else if (env_type[i] == 2)
            n_type_2 += 1.;
        else if (env_type[i] == 3)
            n_type_3 += 1.;
    }

    std::cout << "Environment 1 (circle): " << std::to_string(n_type_1 / num_envs_) << "\n";
    std::cout << "Environment 2 (box): " << std::to_string(n_type_2 / num_envs_) << "\n";
    std::cout << "Environment 3 (corridor): " << std::to_string(n_type_3 / num_envs_) << "\n";

    environments_.reserve(num_envs_);
    rewardInformation_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
        environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, env_type[i], seed_seq[i], obstacle_grid_size_seq[i], obstacle_dr_seq[i]));
        environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
        environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
        rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  // resets specific environments and returns observation
  void partial_reset(Eigen::Ref<EigenBoolVec> &needed_reset) {
      for (int i = 0; i < num_envs_; i++)
          if (needed_reset[i])
              environments_[i]->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
  }

  void coordinate_observe(Eigen::Ref<EigenRowMajorMat> &coordinate) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->coordinate_observe(coordinate.row(i));
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void partial_step(Eigen::Ref<EigenRowMajorMat> &action,
                    Eigen::Ref<EigenVec> &reward,
                    Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          if (done[i] == false)
              perAgentStep(i, action, reward, done);
  }

  void set_goal(Eigen::Ref<EigenVec> &goal) { environments_[0]->set_goal(goal); }

  void baseline_compute_reward(Eigen::Ref<EigenRowMajorMat> &sampled_command,
                               Eigen::Ref<EigenVec> &goal_Pos_local,
                               Eigen::Ref<EigenVec> &rewards_p,
                               Eigen::Ref<EigenVec> &collision_idx,
                               int steps, double delta_t, double must_safe_time) {
      environments_[0]->baseline_compute_reward(sampled_command, goal_Pos_local, rewards_p, collision_idx,
                                                steps, delta_t, must_safe_time);
  }

  void computed_heading_direction(Eigen::Ref<EigenVec> &heading_direction) {
      environments_[0]->computed_heading_direction(heading_direction);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void visualize_desired_command_traj(Eigen::Ref<EigenRowMajorMat> &coordinate_desired_command,
                                      Eigen::Ref<EigenVec> &P_col_desired_command) {
      environments_[0]->visualize_desired_command_traj(coordinate_desired_command, P_col_desired_command);
  }

  void visualize_modified_command_traj(Eigen::Ref<EigenRowMajorMat> &coordinate_modified_command,
                                       Eigen::Ref<EigenVec> &P_col_modified_command) {
      environments_[0]->visualize_modified_command_traj(coordinate_modified_command, P_col_modified_command);
  }

  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  void reward_logging(Eigen::Ref<EigenRowMajorMat> &rewards) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->reward_logging(rewards.row(i));
  }

  void contact_logging(Eigen::Ref<EigenRowMajorMat> &contacts) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
        environments_[i]->contact_logging(contacts.row(i));
    }

    void torque_and_velocity_logging(Eigen::Ref<EigenRowMajorMat> &torque_and_velocity) {
  #pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->torque_and_velocity_logging(torque_and_velocity.row(i));
    }

  void set_user_command(Eigen::Ref<EigenRowMajorMat> &command) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->set_user_command(command.row(i));
  }

  void initialize_n_step() {
      for (auto *env: environments_)
          env->initialize_n_step();
  }

  const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

//    rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

    float terminalReward = 0.0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

//    if (done[agentId]) {
//      environments_[agentId]->reset();
//      reward[agentId] += terminalReward;
//    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::map<std::string, float>> rewardInformation_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
