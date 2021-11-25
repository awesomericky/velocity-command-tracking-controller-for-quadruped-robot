//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

// [Tip]
//
// // Logging example
// Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
// std::cout << current_leg_phase.format(CommaInitFmt) << std::endl;
//std::cout << obstacle_distance_cost << "\n";
//
// // To make new function
// 1. Environment.hpp
// 2. raisim_gym.cpp (if needed)
// 3. RaisimGymEnv.hpp
// 4. VectorizedEnvironment.hpp
// 5. RaisimGymVecEnv.py (if needed)

namespace raisim
{

    class ENVIRONMENT : public RaisimGymEnv
    {

    public:
        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int sample_env_type, int seed, double sample_obstacle_grid_size, double sample_obstacle_dr)
        : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable)
        {

            /// create world
            world_ = std::make_unique<raisim::World>();
            world_type = cfg["type"].As<int>();

            /// If .... ==> generate env
            if (world_type == 1)
                generate_env_1();
            else if (world_type == 2)
                generate_env_2();
            else if (world_type == 3)
                generate_env_3(seed, sample_obstacle_grid_size, sample_obstacle_dr);
            else if (world_type == 5)
                generate_env_5(seed);
            else if (world_type == 6)
                generate_env_6();
            else if (world_type == 7)
                generate_env_7();
            else if (world_type == 8)
                generate_env_8();
            else if (world_type == 9)
                generate_env_9();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_ + "/anymal_c/urdf/anymal.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            /// add box for contact checking for anymal
            anymal_box_ = world_->addBox(1.1, 0.55, 0.2, 0.1, "default", raisim::COLLISION(2), RAISIM_STATIC_COLLISION_GROUP);
            anymal_box_->setName("anymal_box");

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim(); // 3(base position) + 4(base orientation) + 12(joint position) = 19
            gvDim_ = anymal_->getDOF();                      // 3(base linear velocity) + 3(base angular velocity) + 12(joint velocity) = 18
            nJoints_ = gvDim_ - 6;                           // 12

            /// set depth sensor (lidar)
            lidar_theta = M_PI * 2;
            delta_lidar_theta = M_PI / 180;
            scanSize = int(lidar_theta / delta_lidar_theta);

            /// initialize containers
            gc_.setZero(gcDim_);
            gc_init_.setZero(gcDim_);
            random_gc_init.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            random_gv_init.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);
            joint_position_error_history.setZero(nJoints_ * n_history_steps);
            joint_velocity_history.setZero(nJoints_ * n_history_steps);
            GRF_impulse.setZero(4);

            /// Add intialization for extra cost terms
            previous_action.setZero(nJoints_);
            target_postion.setZero(nJoints_);
            footPos_W.resize(4);
            footVel_W.resize(4);
            footContactVel_.resize(4);

            /// Initialize user command values
            before_user_command.setZero(3);
            after_user_command.setZero(3);

            /// collect joint positions, collision geometry
            defaultJointPositions_.resize(13);
            defaultBodyMasses_.resize(13);
            for (int i = 0; i < 13; i++) {
                defaultJointPositions_[i] = anymal_->getJointPos_P()[i].e();
                defaultBodyMasses_[i] = anymal_->getMass(i);
            }

            /// Get COM_base position
            COMPosition_ = anymal_->getBodyCOM_B()[0].e();

            /// reward weights
            reward_obstacle_distance_coeff = cfg["reward"]["obstacle_distance"]["coeff"].As<double>();
            reward_command_similarity_coeff = cfg["reward"]["command_similarity"]["coeff"].As<double>();

            /// total trajectory length
            double control_dt = cfg["control_dt"].As<double>();
            double max_time = cfg["max_time"].As<double>();
            double command_period = cfg["command_period"].As<double>();
            total_traj_len = int(max_time / control_dt);
            command_len = int(command_period / control_dt);

            /// Randomization
            randomization = cfg["randomization"].template As<bool>();
            if (randomization) {
                /// Randomize mass and Dynamics (joint position)
                noisify_Dynamics();
                noisify_Mass_and_COM();
            }
            random_initialize = cfg["random_initialize"].template As<bool>();
            random_external_force = cfg["random_external_force"].template As<bool>();

            /// contact foot index
            foot_idx[0] = anymal_->getFrameIdxByName("LF_shank_fixed_LF_FOOT");  // 3
            foot_idx[1] = anymal_->getFrameIdxByName("RF_shank_fixed_RF_FOOT");  // 6
            foot_idx[2] = anymal_->getFrameIdxByName("LH_shank_fixed_LH_FOOT");  // 9
            foot_idx[3] = anymal_->getFrameIdxByName("RH_shank_fixed_RH_FOOT");  // 12

            /// contact shank index
            shank_idx[0] = anymal_->getFrameIdxByName("LF_KFE");
            shank_idx[1] = anymal_->getFrameIdxByName("RF_KFE");
            shank_idx[2] = anymal_->getFrameIdxByName("LH_KFE");
            shank_idx[3] = anymal_->getFrameIdxByName("RH_KFE");

            /// nominal configuration of anymal_c
            if (world_type == 5)
                gc_init_ << -3, 0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.03, 0.5, -0.9, -0.03, 0.5, -0.9, 0.03, -0.5, 0.9, -0.03, -0.5, 0.9;
            else if (world_type == 2) {
                gc_init_ << hm_sizeX - 3.7, hm_sizeY - 0.8, 0.7, 1.0, 0.0, 0.0, 0.0, 0.03, 0.5, -0.9, -0.03, 0.5, -0.9, 0.03, -0.5, 0.9, -0.03, -0.5, 0.9;  //0.5

                raisim::Vec<3> axis;
                raisim::Vec<4> quaternion;

                // orientation
                axis[0] = 0;
                axis[1] = 0;
                axis[2] = 1;
                double angle = - 0.5 * M_PI;
                raisim::angleAxisToQuaternion(axis, angle, quaternion);
                gc_init_.segment(3, 4) = quaternion.e();
            }
            else
                gc_init_ << 0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.03, 0.5, -0.9, -0.03, 0.5, -0.9, 0.03, -0.5, 0.9, -0.03, -0.5, 0.9;  //0.5
            random_gc_init = gc_init_; random_gv_init = gv_init_;

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero();
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero();
            jointDgain.tail(nJoints_).setConstant(0.2);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 81 + scanSize;
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);
            coordinateDouble.setZero(3);

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            actionStd_.setConstant(0.3);

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("LF_THIGH"));
            footIndices_.insert(anymal_->getBodyIdx("RF_THIGH"));
            footIndices_.insert(anymal_->getBodyIdx("LH_THIGH"));
            footIndices_.insert(anymal_->getBodyIdx("RH_THIGH"));

            /// visualize if it is the first environment
            if (visualizable_)
            {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();

                /// visualize scans
                for (int i=0; i<scanSize; i++)
                    scans.push_back(server_->addVisualBox("box" + std::to_string(i), 0.05, 0.05, 0.05, 1, 0, 0));

                /// visualize scans
                for (int i=0; i<n_prediction_step; i++) {
                    desired_command_traj.push_back(server_->addVisualBox("desired_command_pos" + std::to_string(i), 0.08, 0.08, 0.08, 1, 1, 0));
                    modified_command_traj.push_back(server_->addVisualBox("modified_command_pos" + std::to_string(i), 0.08, 0.08, 0.08, 0, 1, 1));
                }

//                /// visualize future position
//                for (int i=0; i<12; i++) {
//                    prediction_box_1.push_back(server_->addVisualBox("prediction1_" + std::to_string(i), 1.1, 0.2, 0.55, 1, 0, 0));
//                    prediction_box_2.push_back(server_->addVisualBox("prediction2_" + std::to_string(i), 1.1, 0.2, 0.55, 0, 1, 0));
//                }

                /// goal
                if (world_type == 2) {
                    server_->addVisualCylinder("goal1", 0.4, 0.8, 2, 1, 0);
                    server_->addVisualCylinder("goal2", 0.4, 0.8, 2, 1, 0);
                    server_->addVisualCylinder("goal3", 0.4, 0.8, 2, 1, 0);
                }
                else {
                    server_->addVisualCylinder("goal", 0.4, 0.8, 2, 1, 0);
                }

//                /// set path
//                double max_time = 120., delta_time = 0.05;
//                int max_num_path_slice = int(max_time / delta_time);
//                for (int i=0; i<max_num_path_slice; i++)
//                    server_->addVisualBox("path" + std::to_string(i+1), 0.1, 0.1, 0.1, 1, 0, 0);

                server_->focusOn(anymal_);
            }
        }

    void baseline_compute_reward(Eigen::Ref<EigenRowMajorMat> sampled_command,
                                 Eigen::Ref<EigenVec> goal_Pos_local,
                                 Eigen::Ref<EigenVec> rewards_p,
                                 Eigen::Ref<EigenVec> collision_idx,
                                 int steps, double delta_t, double must_safe_time)
    {
        int n_sample = sampled_command.rows();
        raisim::Vec<3> future_coordinate;
        raisim::Vec<4> future_quaternion;
        Eigen::VectorXd rewards_cp;
        Eigen::VectorXd collision_idx_cp;
        int must_safe_n_steps = int(must_safe_time / delta_t);

        rewards_cp.setZero(n_sample);
        collision_idx_cp.setZero(n_sample);
        double current_goal_distance = goal_Pos_local.norm();

        if (server_)
            server_->lockVisualizationServerMutex();

        for (int i=0; i<n_sample; i++) {
            double local_x = 0;
            double local_y = 0;
            double local_yaw = 0;
            for (int j=0; j<steps; j++) {
                local_yaw += sampled_command(i, 2) * delta_t;
                local_x += sampled_command(i, 0) * delta_t * cos(local_yaw) - sampled_command(i, 1) * delta_t * sin(local_yaw);
                local_y += sampled_command(i, 0) * delta_t * sin(local_yaw) + sampled_command(i, 1) * delta_t * cos(local_yaw);
                future_coordinate[0] = local_x * cos(coordinateDouble[2]) - local_y * sin(coordinateDouble[2]) + coordinateDouble[0];
                future_coordinate[1] = local_x * sin(coordinateDouble[2]) + local_y * cos(coordinateDouble[2]) + coordinateDouble[1];
                future_coordinate[2] = local_yaw + coordinateDouble[2];

                raisim::angleAxisToQuaternion({0, 0, 1}, future_coordinate[2], future_quaternion);

//                if (i == 0) {
//                    prediction_box_1[j]->setPosition(future_coordinate[0], future_coordinate[1], 0.5);
//                    prediction_box_1[j]->setOrientation({future_quaternion[0], future_quaternion[1], future_quaternion[2], future_quaternion[3]});
//                }
//                else if (i == 1) {
//                    prediction_box_2[j]->setPosition(future_coordinate[0], future_coordinate[1], 0.5);
//                    prediction_box_2[j]->setOrientation({future_quaternion[0], future_quaternion[1], future_quaternion[2], future_quaternion[3]});
//                }

                if (j < must_safe_n_steps) {
                    anymal_box_->setPosition(future_coordinate[0], future_coordinate[1], 0.5);
                    anymal_box_->setOrientation(future_quaternion);

                    world_->integrate1();

                    int num_anymal_future_contact = anymal_box_->getContacts().size();
                    if (num_anymal_future_contact > 0) {
                        collision_idx_cp[i] = 1;
                    }

//                    if (future_coordinate[0] < -2 || hm_sizeX - 2 < future_coordinate[0] ||
//                        future_coordinate[1] < 0 || hm_sizeY < future_coordinate[1]) {
//                        collision_idx_cp[i] = 1;
//                    }
                }

                double future_goal_distance = sqrt(pow(local_x - goal_Pos_local[0], 2) + pow(local_y - goal_Pos_local[1], 2));
                rewards_cp[i] += current_goal_distance - future_goal_distance;
            }
        }

        if (server_)
            server_->unlockVisualizationServerMutex();

        rewards_p = rewards_cp.cast<float>();
        collision_idx = collision_idx_cp.cast<float>();
    }

    void init() final {}

    void generate_env_1()
    {
        world_->addGround();
        double H1 = 2;
        double H2 = 4;
        double x = H2 - H1;
        double thickness = 0.05;

        raisim::Vec<3> axis;
        raisim::Vec<4> quaternion;
        axis[0] = 0;
        axis[1] = 0;
        axis[2] = 1;
        double angle = 0.5 * M_PI;
        raisim::angleAxisToQuaternion(axis, angle, quaternion);

        auto box1 = world_->addBox(H2, thickness, 1, 1);
        auto box2 = world_->addBox(H2, thickness, 1, 1);
        auto box3 = world_->addBox(H1, thickness, 1, 1);
        auto box4 = world_->addBox(H2, thickness, 1, 1);
        auto box5 = world_->addBox(H2, thickness, 1, 1);
        auto box6 = world_->addBox(H1, thickness, 1, 1);
        box1->setPosition(H2/2, - x/2 - thickness/2, 0.5);
        box2->setPosition(H2 + thickness/2, H2/2 - x/2 - thickness, 0.5);
        box2 ->setOrientation(quaternion);
        box3->setPosition(H2 + H1/2, H2 -x/2 - thickness/2, 0.5);
        box4->setPosition(H1 + H2/2, H2 + x/2 + thickness/2, 0.5);
        box5->setPosition(H1 - thickness/2, x/2 + H2/2 + thickness, 0.5);
        box5->setOrientation(quaternion);
        box6->setPosition(H1/2, x/2 + thickness/2, 0.5);
    }

    void generate_env_2()
    {
        hm_sizeX = 20;
        hm_sizeY = 20;
        auto heightmap = world_->addHeightMap("/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/heightmap/env1.png", hm_sizeX/2 - 2, hm_sizeY/2, hm_sizeX, hm_sizeY, 0.005, 0.0);
        auto heightmap_raw = heightmap->getHeightMap();
        int hm_samplesX = std::sqrt(heightmap_raw.size());
        int hm_samplesY = std::sqrt(heightmap_raw.size());
        double unitX = hm_sizeX / hm_samplesX;
        double unitY = hm_sizeY / hm_samplesY;
        std::vector<raisim::Vec<2>> obstacle_XYs;

        for (int j=0; j<hm_samplesY; j++) {
            for (int i=0; i<hm_samplesX; i++) {
                double x = (i - (hm_samplesX / 2)) * unitX;
                double y = (j - (hm_samplesY / 2)) * unitY;
                int idx = i + hm_samplesX * j;
                if (heightmap_raw[idx] == 0.0) {
                    if ( (- hm_sizeX/2 + 1) < x && x < (hm_sizeX/2 - 1) && (- hm_sizeY/2 + 1) < y && y < (hm_sizeY/2 - 1) )
                        init_set.push_back({x, y});
                }
            }
        }

        n_init_set = init_set.size();

        assert(n_init_set > 0);
    }

    void generate_env_6()
    {
        hm_sizeX = 12;
        hm_sizeY = 7;
        auto heightmap = world_->addHeightMap("/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/heightmap/env2.png", 0, 0, hm_sizeX, hm_sizeY, 0.005, 0.0);
    }

    void generate_env_7()
    {
//        hm_sizeX = 10;
//        hm_sizeY = 10;
        hm_sizeX = 8;
        hm_sizeY = 8;
        auto heightmap = world_->addHeightMap("/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/heightmap/env3.png", 0, 0, hm_sizeX, hm_sizeY, 0.005, 0.0);
    }

    void generate_env_8()
    {
        hm_sizeX = 6;
        hm_sizeY = 10;
        auto heightmap = world_->addHeightMap("/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/heightmap/env4.png", 0, 0, hm_sizeX, hm_sizeY, 0.005, 0.0);
    }

    void generate_env_9()
    {
        hm_sizeX = 8;
        hm_sizeY = 4;
        auto heightmap = world_->addHeightMap("/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/heightmap/env5.png", 0, 0, hm_sizeX, hm_sizeY, 0.005, 0.0);
    }

    void generate_env_3(int seed, double sample_obstacle_grid_size, double sample_obstacle_dr)
    {
        double hm_centerX = 0.0, hm_centerY = 0.0;
        hm_sizeX = 40., hm_sizeY = 40.;
        double hm_samplesX = hm_sizeX * 5, hm_samplesY = hm_sizeY * 5;
        double unitX = hm_sizeX / hm_samplesX, unitY = hm_sizeY / hm_samplesY;
        double obstacle_height = 2;

        random_seed = seed;
        static std::default_random_engine env_generator(random_seed);

        /// sample obstacle center
        double obstacle_grid_size = sample_obstacle_grid_size;
        obstacle_dr = sample_obstacle_dr;
        int n_x_grid = int(hm_sizeX / obstacle_grid_size);
        int n_y_grid = int(hm_sizeY / obstacle_grid_size);
        n_obstacle = n_x_grid * n_y_grid;

        obstacle_centers.setZero(n_obstacle, 2);
        std::uniform_real_distribution<> uniform(0, obstacle_grid_size - 0.5);
        for (int i=0; i<n_obstacle; i++) {
            int current_n_y = int(i / n_x_grid);
            int current_n_x = i - current_n_y * n_x_grid;
            double sampled_x = uniform(env_generator);
            double sampled_y = uniform(env_generator);
            sampled_x +=  obstacle_grid_size * current_n_x;
            sampled_x -= hm_sizeX/2;
            sampled_y += obstacle_grid_size * current_n_y;
            sampled_y -= hm_sizeY/2;
            obstacle_centers(i, 0) = sampled_x;
            obstacle_centers(i, 1) = sampled_y;
        }

        /// generate obstacles
        for (int j=0; j<hm_samplesY; j++) {
            for (int i=0; i<hm_samplesX; i++) {
                double x = (i - (hm_samplesX / 2)) * unitX, y = (j - (hm_samplesY / 2)) * unitY;
                bool available_obstacle = false, not_available_init = false;
                //                    double min_distance = 100.;
                for (int k=0; k<n_obstacle; k++) {
                    double obstacle_x = obstacle_centers(k, 0), obstacle_y = obstacle_centers(k, 1);
                    if (sqrt(pow(x - obstacle_x, 2) + pow(y - obstacle_y, 2)) < obstacle_dr)
                        available_obstacle = true;
                    if (sqrt(pow(x - obstacle_x, 2) + pow(y - obstacle_y, 2)) < (obstacle_dr + 0.7))
                        not_available_init = true;
                }

                if (j==0 || j==hm_samplesY-1)
                    available_obstacle = true;
                if (i==0 || i==hm_samplesX-1)
                    available_obstacle = true;

                if ( x < (- hm_sizeX/2 + 3) || (hm_sizeX/2 - 3) < x ||
                y < (- hm_sizeY/2 + 3) || (hm_sizeY/2 - 3) < y)
                    not_available_init = true;

                if (!available_obstacle)
                    hm_raw_value.push_back(0.0);
                else
                    hm_raw_value.push_back(obstacle_height);

                if (!not_available_init)
                    init_set.push_back({x, y});
            }
        }

        n_init_set = init_set.size();

        assert(n_init_set > 0);

        /// add heightmap
        hm = world_->addHeightMap(hm_samplesX, hm_samplesY, hm_sizeX, hm_sizeY, hm_centerX, hm_centerY, hm_raw_value);
    }

    void generate_env_5(int seed)
    {
        obstacle_dr = 0.3;
        double hm_centerX = 0.0, hm_centerY = 0.0;
        hm_sizeX = 40., hm_sizeY = 40.;
        double hm_samplesX = hm_sizeX * 3, hm_samplesY = hm_sizeY * 3;
        double unitX = hm_sizeX / hm_samplesX, unitY = hm_sizeY / hm_samplesY;
        double obstacle_height = 2;

        random_seed = seed;
        static std::default_random_engine env_generator(random_seed);

        /// sample obstacle center
        double obstacle_grid_size = 4;
        int n_x_grid = int(hm_sizeX / obstacle_grid_size);
        int n_y_grid = int(hm_sizeY / obstacle_grid_size);
        n_obstacle = n_x_grid * n_y_grid;

        obstacle_centers.setZero(2, 2);
        double obstacle_distance_wo_r = 1.2;
        double obstacle_distance = 2 * obstacle_dr + obstacle_distance_wo_r;
        obstacle_centers(0, 1) = - obstacle_distance / 2;
        obstacle_centers(1, 1) = obstacle_distance / 2;

        /// generate obstacles
        for (int j=0; j<hm_samplesY; j++) {
            for (int i=0; i<hm_samplesX; i++) {
                double x = (i - (hm_samplesX / 2)) * unitX, y = (j - (hm_samplesY / 2)) * unitY;
                bool available_obstacle = false, not_available_init = false;
                double min_distance = 100.;
                for (int k=0; k<2; k++) {
                    double obstacle_x = obstacle_centers(k, 0), obstacle_y = obstacle_centers(k, 1);
                    if (sqrt(pow(x - obstacle_x, 2) + pow(y - obstacle_y, 2)) < obstacle_dr)
                        available_obstacle = true;
                    if (sqrt(pow(x - obstacle_x, 2) + pow(y - obstacle_y, 2)) < (obstacle_dr + 1))
                        not_available_init = true;
                }

                if (j==0 || j==hm_samplesY-1)
                    available_obstacle = true;
                if (i==0 || i==hm_samplesX-1)
                    available_obstacle = true;

                if ( x < (- hm_sizeX/2 + 3) || (hm_sizeX/2 - 3) < x ||
                y < (- hm_sizeY/2 + 3) || (hm_sizeY/2 - 3) < y)
                    not_available_init = true;

                assert(available_obstacle && !not_available_init);

                if (!available_obstacle)
                    hm_raw_value.push_back(0.0);
                else
                    hm_raw_value.push_back(obstacle_height);

                if (!not_available_init)
                    init_set.push_back({x, y});
            }
        }

        n_init_set = init_set.size();

        assert(n_init_set > 0);

        /// add heightmap
        hm = world_->addHeightMap(hm_samplesX, hm_samplesY, hm_sizeX, hm_sizeY, hm_centerX, hm_centerY, hm_raw_value);
    }

    void reset() final
    {
        static std::default_random_engine generator(random_seed);

        if (random_initialize) {
            if (current_n_step == 0) {
                /// Random initialization by sampling available x, y position
                std::uniform_int_distribution<> uniform_init(0, n_init_set-1);
                std::normal_distribution<> normal(0, 1);
                int n_init = uniform_init(generator);
                raisim::Vec<3> random_axis;
                raisim::Vec<4> random_quaternion;

                // Random position
                for (int i=0; i<2; i++)
                    random_gc_init[i] = init_set[n_init][i];

                // Random orientation (just randomizing yaw angle)
                random_axis[0] = 0;
                random_axis[1] = 0;
                random_axis[2] = 1;
                std::uniform_real_distribution<> uniform_angle(-1, 1);
                double random_angle = uniform_angle(generator) * M_PI;
                raisim::angleAxisToQuaternion(random_axis, random_angle, random_quaternion);
                random_gc_init.segment(3, 4) = random_quaternion.e();

                current_random_gc_init = random_gc_init;
                current_random_gv_init = random_gv_init;
                anymal_->setState(random_gc_init, random_gv_init);
                initHistory();
            }
            else {
                anymal_->setState(current_random_gc_init, current_random_gv_init);
                initHistory();
            }

        }
        else {
            anymal_->setState(gc_init_, gv_init_);
            initHistory();
        }

        if (world_type == 6) {
            raisim::Vec<4> quaternion;
            raisim::angleAxisToQuaternion({0, 0, 1}, M_PI/2, quaternion);

            gc_init_[0] = 3.3;
            gc_init_[1] = 0.3;
            gc_init_.segment(3, 4) = quaternion.e();
            anymal_->setState(gc_init_, gv_init_);
            initHistory();
        }
        else if (world_type == 7) {
            raisim::Vec<4> quaternion;
            gc_init_[0] = -2;
//            gc_init_[1] = - 1.5;
            gc_init_[1] = - 1.;
            anymal_->setState(gc_init_, gv_init_);
            initHistory();
        }
        else if (world_type == 8) {
            raisim::Vec<4> quaternion;
            raisim::angleAxisToQuaternion({0, 0, 1}, M_PI/2, quaternion);
            gc_init_[0] = 0;
            gc_init_[1] = 4.;
            gc_init_.segment(3, 4) = quaternion.e();
            anymal_->setState(gc_init_, gv_init_);
            initHistory();
        }
        else if (world_type == 9) {
//            raisim::Vec<4> quaternion;
//            raisim::angleAxisToQuaternion({0, 0, 1}, M_PI/2, quaternion);
            gc_init_[0] = 3;
            gc_init_[1] = -1.;
//            gc_init_.segment(3, 4) = quaternion.e();
            anymal_->setState(gc_init_, gv_init_);
            initHistory();
        }

        if (random_external_force) {
            random_force_period = int(1.0 / control_dt_);
            std::uniform_int_distribution<> uniform_force(1, total_traj_len - random_force_period);
            std::uniform_int_distribution<> uniform_binary(0, 1);
            random_force_n_step = uniform_force(generator);
            random_external_force_final = uniform_binary(generator);  /// 0: x, 1: o
            random_external_force_direction = uniform_binary(generator);  /// 0: -1, 1: +1
        }

        updateObservation();
    }

    float step(const Eigen::Ref<EigenVec> &action) final
    {
        current_n_step += 1;

        /// action scaling
        pTarget12_ = action.cast<double>();
        pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
        pTarget12_ += actionMean_;
        target_postion = pTarget12_.cast<double>();
        pTarget_.tail(nJoints_) = pTarget12_;

        Eigen::VectorXd current_joint_position_error = pTarget12_ - gc_.tail(nJoints_);
        updateHistory(current_joint_position_error, gv_.tail(nJoints_));

        anymal_->setPdTarget(pTarget_, vTarget_);

        /// Set external force to the base of the robot
        if (random_external_force)
            if (bool (random_external_force_final))
                if (random_force_n_step <= current_n_step && current_n_step < (random_force_n_step + random_force_period)) {
                    raisim::Mat<3, 3> baseOri;
                    Eigen::Vector3d force_direction;
                    anymal_->getFrameOrientation("base_to_base_inertia", baseOri);
                    if (random_external_force_direction == 0)
                        force_direction = {0, -1, 0};
                    else
                        force_direction = {0, 1, 0};
                    force_direction = baseOri.e() * force_direction;
                    anymal_->setExternalForce(anymal_->getBodyIdx("base"), force_direction * 50);
                }

        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
        {
            if (server_)
                server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_)
                server_->unlockVisualizationServerMutex();
        }

        updateObservation();

        torque = anymal_->getGeneralizedForce().e(); // squaredNorm

        calculate_cost();

        rewards_.record("obstacle_distance", -obstacle_distance_cost);
        rewards_.record("command_similarity", -command_similarity_cost);

        previous_action = target_postion.cast<double>();

        return rewards_.sum();
    }

    void updateObservation()
    {
        anymal_->getState(gc_, gv_);
        raisim::Vec<4> quat;
        raisim::Mat<3, 3> rot;
        quat[0] = gc_[3];
        quat[1] = gc_[4];
        quat[2] = gc_[5];
        quat[3] = gc_[6];
        raisim::quatToRotMat(quat, rot);
        bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
        bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

//        /// Visualize base path
//        if (current_n_step != 0 && current_n_step % 5 == 0)
//            if (visualizable_) {
//                server_->getVisualObject("path" + std::to_string(current_n_step/5))->setPosition(gc_.segment(0, 3));
//            }

        /// Get depth data
        raisim::Vec<3> lidarPos;
        raisim::Mat<3,3> lidarOri;
        anymal_->getFramePosition("lidar_cage_to_lidar", lidarPos);
        anymal_->getFrameOrientation("lidar_cage_to_lidar", lidarOri);
        int ray_length = 10;
        Eigen::Vector3d direction;
        Eigen::Vector3d rayDirection;

        lidar_scan_depth.setOnes(scanSize);

        for (int i=0; i<scanSize; i++) {
            const double angle = - lidar_theta / 4 + delta_lidar_theta * i;
            direction = {cos(angle), sin(angle), 0};
            rayDirection = lidarOri.e() * direction;

            /// front lidar
            auto &col = world_->rayTest(lidarPos.e(), rayDirection, ray_length, true);
            if (col.size() > 0) {
//                if (visualizable_)
//                    scans[i]->setPosition(col[0].getPosition());
                lidar_scan_depth[i] = (lidarPos.e() - col[0].getPosition()).norm() / ray_length;
            }
            else {
//                if (visualizable_)
//                    scans[i]->setPosition({0, 0, 100});
            }
        }

        /// transformed user command should be concatenated to the observation
        obDouble_ << rot.e().row(2).transpose(),      /// body orientation (dim=3)
                gc_.tail(12),                    /// joint angles (dim=12)
                bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity (dim=3+3=6)
                gv_.tail(12),                  /// joint velocity (dim=12)
                joint_position_error_history,    /// joint position error history (dim=24)
                joint_velocity_history,          /// joint velocity history (dim=24)
                lidar_scan_depth;                /// Lidar scan data (normalized)
        
        /// Update coordinate
        double yaw = atan2(rot.e().col(0)[1], rot.e().col(0)[0]);

        coordinateDouble << gc_[0], gc_[1], yaw;

        roll_and_pitch = rot.e().row(2).transpose();
        GRF_impulse.setZero(4);

//        comprehend_contacts();
    }

    void updateHistory(Eigen::VectorXd current_joint_position_error,
                       Eigen::VectorXd current_joint_velocity)
    {
        /// 0 ~ 11 : t-2 step
        /// 12 ~ 23 : t-1 step
        for (int i; i<n_history_steps-1; i++) {
            joint_position_error_history.segment(i * nJoints_, nJoints_) = joint_position_error_history.segment((i+1) * nJoints_, nJoints_);
            joint_velocity_history.segment(i * nJoints_, nJoints_) = joint_velocity_history.segment((i+1) * nJoints_, nJoints_);
        }
        joint_position_error_history.tail(nJoints_) = current_joint_position_error;
        joint_velocity_history.tail(nJoints_) = current_joint_velocity;
    }

    void initHistory()
    {
        joint_position_error_history.setZero(nJoints_ * n_history_steps);
        joint_velocity_history.setZero(nJoints_ * n_history_steps);
    }

    void observe(Eigen::Ref<EigenVec> ob) final
    {
        /// convert it to float
        /// 0 - 80 : proprioceptive sensor data
        /// 81 - : lidar sensor data
        ob = obDouble_.cast<float>();
    }

    void coordinate_observe(Eigen::Ref<EigenVec> coordinate)
    {
        /// convert it to float
        coordinate = coordinateDouble.cast<float>();
    }

    void calculate_cost()
    {
        /// New rewards for collision avoidance
        obstacle_distance_cost = 0.0, command_similarity_cost = 0.0;
    }

    void comprehend_contacts()
    {
        numContact_ = anymal_->getContacts().size();

        Eigen::Vector4d foot_Pos_height_map;
        float shank_dr = 0.06;

        for (int k = 0; k < 4; k++)
        {
            footContactState_[k] = false;
            anymal_->getFramePosition(foot_idx[k], footPos_W[k]);  //position of the feet
            anymal_->getFrameVelocity(foot_idx[k], footVel_W[k]);
            foot_Pos_height_map[k] = hm->getHeight(footPos_W[k][0], footPos_W[k][1]);
//            foot_Pos_height_map[k] = 0.;   /// Should change if it is rough terrain!!!!
            foot_Pos_difference[k] = std::abs(footPos_W[k][2] - foot_Pos_height_map[k]);

            std::vector<raisim::Vec<3>> shankPos_W;
            shankPos_W.resize(4);
            anymal_->getFramePosition(shank_idx[k], shankPos_W[k]);
            shank_Pos_difference[k] = std::abs(shankPos_W[k][2] - shank_dr);
        }

        raisim::Vec<3> vec3;
        float dr = 0.03;

        //Classify foot contact
        /// This only works for flat terrain!!
        if (numContact_ > 0)
        {
            for (int k = 0; k < numContact_; k++)
            {
                if (!anymal_->getContacts()[k].skip())
                {
                    int idx = anymal_->getContacts()[k].getlocalBodyIndex();

                    // check foot height to distinguish shank contact
                    if (idx == 3 && foot_Pos_difference[0] <= dr && !footContactState_[0])
                    {
                        footContactState_[0] = true;
                        // footNormal_[0] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[0] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[0] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 6 && foot_Pos_difference[1] <= dr && !footContactState_[1])
                    {
                        footContactState_[1] = true;
                        // footNormal_[1] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[1] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[1] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 9 && foot_Pos_difference[2] <= dr && !footContactState_[2])
                    {
                        footContactState_[2] = true;
                        // footNormal_[2] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[2] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[2] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 12 && foot_Pos_difference[3] <= dr && !footContactState_[3])
                    {
                        footContactState_[3] = true;
                        // footNormal_[3] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[3] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[3] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                }
            }
        }
    }

    void visualize_desired_command_traj(Eigen::Ref<EigenRowMajorMat> coordinate_desired_command,
                                        Eigen::Ref<EigenVec> P_col_desired_command,
                                        double collision_threshold)
    {
        for (int i=0; i<n_prediction_step; i++) {
            if (P_col_desired_command[i] < collision_threshold) {
                /// not collide
                desired_command_traj[i]->setColor(1, 1, 0, 1);  // yellow
            }
            else {
                /// collide
                desired_command_traj[i]->setColor(1, 0, 0, 1);  // red
            }
            const double coordinate_z = 0;
            desired_command_traj[i]->setPosition({coordinate_desired_command(i, 0), coordinate_desired_command(i, 1), coordinate_z});

            /// reset modified command trajectory
            modified_command_traj[i]->setPosition({0, 0, 100});
        }
    }

    void visualize_modified_command_traj(Eigen::Ref<EigenRowMajorMat> coordinate_modified_command,
                                         Eigen::Ref<EigenVec> P_col_modified_command,
                                         double collision_threshold)
    {
        for (int i=0; i<n_prediction_step; i++) {
            if (P_col_modified_command[i] < collision_threshold) {
                /// not collide
                modified_command_traj[i]->setColor(0, 0, 1, 1);  // blue
            }
            else {
                /// collide
                modified_command_traj[i]->setColor(1, 0, 0, 1);  // red
            }
            const double coordinate_z = 0;
            modified_command_traj[i]->setPosition({coordinate_modified_command(i, 0), coordinate_modified_command(i, 1), coordinate_z});
        }
    }

    void set_user_command(Eigen::Ref<EigenVec> command) {}

    void reward_logging(Eigen::Ref<EigenVec> rewards, Eigen::Ref<EigenVec> rewards_w_coeff, int n_rewards) {}

    void noisify_Dynamics() {
        static std::default_random_engine generator(random_seed);
        std::uniform_real_distribution<> uniform01(0.0, 1.0);
        std::uniform_real_distribution<> uniform(-1.0, 1.0);

        /// joint position randomization
        for (int i = 0; i < 4; i++) {
            double x_, y_, z_;
            if (i < 2) x_ = uniform01(generator) * 0.005;
            else x_ = -uniform01(generator) * 0.005;

            y_ = uniform(generator) * 0.01;
            z_ = uniform(generator) * 0.01;

            int hipIdx = 3 * i + 1;
            int thighIdx = 3 * i + 2;
            int shankIdx = 3 * i + 3;

            ///hip
            anymal_->getJointPos_P()[hipIdx].e()[0] = defaultJointPositions_[hipIdx][0] + x_;
            anymal_->getJointPos_P()[hipIdx].e()[1] = defaultJointPositions_[hipIdx][1] + y_;
            anymal_->getJointPos_P()[hipIdx].e()[2] = defaultJointPositions_[hipIdx][2] + z_; ///1


            /// thigh
            x_ = - uniform01(generator) * 0.01;
            y_ = uniform(generator) * 0.01;
            z_ = uniform(generator) * 0.01;

            anymal_->getJointPos_P()[thighIdx].e()[0] = defaultJointPositions_[thighIdx][0] + x_;
            anymal_->getJointPos_P()[thighIdx].e()[1] = defaultJointPositions_[thighIdx][1] + y_;
            anymal_->getJointPos_P()[thighIdx].e()[2] = defaultJointPositions_[thighIdx][2] + z_; ///1

            /// shank
            double dy_ = uniform(generator) * 0.005;
            //  dy>0 -> move outwards
            if (i % 2 == 1) {
                y_ = -dy_;
            } else {
                y_ = dy_;
            }

            x_ = uniform(generator) * 0.01;
            z_ = uniform(generator) * 0.01;

            anymal_->getJointPos_P()[shankIdx].e()[0] = defaultJointPositions_[shankIdx][0] + x_;
            anymal_->getJointPos_P()[shankIdx].e()[1] = defaultJointPositions_[shankIdx][1] + y_;
            anymal_->getJointPos_P()[shankIdx].e()[2] = defaultJointPositions_[shankIdx][2] + z_;

        }
    }

    void noisify_Mass_and_COM() {
        static std::default_random_engine generator(random_seed);
        std::uniform_real_distribution<> uniform(-1.0, 1.0);

        /// base mass
        anymal_->getMass()[0] = defaultBodyMasses_[0] * (1 + 0.15 * uniform(generator));

        /// hip mass
        for (int i = 1; i < 13; i += 3) {
            anymal_->getMass()[i] = defaultBodyMasses_[i] * (1 + 0.15 * uniform(generator));
        }

        /// thigh mass
        for (int i = 2; i < 13; i += 3) {
            anymal_->getMass()[i] = defaultBodyMasses_[i] * (1 + 0.15 * uniform(generator));
        }

        /// shank mass
        for (int i = 3; i < 13; i += 3) {
            anymal_->getMass()[i] = defaultBodyMasses_[i] * (1 + 0.04 * uniform(generator));
        }

        anymal_->updateMassInfo();

        /// COM position
        for (int i = 0; i < 3; i++) {
            anymal_->getBodyCOM_B()[0].e()[i] = COMPosition_[i] + uniform(generator) * 0.01;
        }
    }

    void contact_logging(Eigen::Ref<EigenVec> contacts)
    {
        contacts = GRF_impulse.cast<float>();
    }

    void torque_and_velocity_logging(Eigen::Ref<EigenVec> torque_and_velocity)
    {
        /// 0 ~ 12 : torque, 12 ~ 24 : joint velocity
        torque_and_velocity.segment(0, 12) = torque.tail(12).cast<float>();
        torque_and_velocity.segment(12, 12) = gv_.segment(6, 12).cast<float>();
    }

    void set_goal(Eigen::Ref<EigenVec> goal_pos)
    {
        if (world_type == 2) {
            std::vector<Eigen::Vector2d> goal_pos_;
            goal_pos_.push_back({hm_sizeX * 5 / 7, hm_sizeY * 1 / 6});
            goal_pos_.push_back({hm_sizeX * 1 / 70, hm_sizeY * 5 / 20});
            goal_pos_.push_back({hm_sizeX * 3 / 7, hm_sizeY - 0.8});
            goal_pos = goal_pos_[num_set_goal].cast<float>();

            server_->getVisualObject("goal1")->setPosition({goal_pos_[0][0], goal_pos_[0][1], 0.05});
            server_->getVisualObject("goal2")->setPosition({goal_pos_[1][0], goal_pos_[1][1], 0.05});
            server_->getVisualObject("goal3")->setPosition({goal_pos_[2][0], goal_pos_[2][1], 0.05});

//            server_->getVisualObject("goal")->setPosition({goal_pos_[num_set_goal][0], goal_pos_[num_set_goal][1], 0.05});

            num_set_goal += 1;
        }
        else if (world_type == 6) {
            Eigen::Vector2d goal_pos_;
            goal_pos_[0] = -3.0;
            goal_pos_[1] = 0.0;
            goal_pos = goal_pos_.cast<float>();

            server_->getVisualObject("goal")->setPosition({goal_pos_[0], goal_pos_[1], 0.05});
        }
        else if (world_type == 7) {
            std::vector<Eigen::Vector2d> goal_pos_;
            goal_pos_.push_back({0, 1.5});
            goal_pos_.push_back({2, -1.});
            goal_pos = goal_pos_[num_set_goal].cast<float>();

            server_->getVisualObject("goal")->setPosition({goal_pos_[num_set_goal][0], goal_pos_[num_set_goal][1], 0.05});

            num_set_goal += 1;
        }
        else if (world_type == 8) {
            Eigen::Vector2d goal_pos_;
            goal_pos_[0] = 0.;
            goal_pos_[1] = -4.;
            goal_pos = goal_pos_.cast<float>();

            server_->getVisualObject("goal")->setPosition({goal_pos_[0], goal_pos_[1], 0.05});
        }
        else if (world_type == 9) {
            Eigen::Vector2d goal_pos_;
            goal_pos_[0] = -3.;
            goal_pos_[1] = 0.75;
            goal_pos = goal_pos_.cast<float>();

            server_->getVisualObject("goal")->setPosition({goal_pos_[0], goal_pos_[1], 0.05});
        }
        else {
            Eigen::VectorXd goal_pos_;
            goal_pos_.setZero(2);
            static std::default_random_engine generator(1000);
            std::uniform_int_distribution<> uniform_init(0, n_init_set-1);
            int n_init = uniform_init(generator);

            // Gaol position
            for (int i=0; i<2; i++)
                goal_pos_[i] = init_set[n_init][i];

            goal_pos = goal_pos_.cast<float>();

            server_->getVisualObject("goal")->setPosition({goal_pos_[0], goal_pos_[1], 0.05});
        }
    }

    void initialize_n_step()
    {
        current_n_step = 0;
    }

    void computed_heading_direction(Eigen::Ref<EigenVec> heading_direction_) {}

    bool isTerminalState(float &terminalReward) final
    {
        terminalReward = float(terminalRewardCoeff_);

        /// if anymal falls down
        raisim::Vec<3> base_position;
        anymal_->getFramePosition("base_to_base_inertia", base_position);
        if (base_position[2] < 0.3)
            return true;

//        /// if the contact body is not feet (this terminal condition includes crashing with obstacle)
//        for (auto &contact : anymal_->getContacts()) {
//            if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
//                return true;
//            }
////            for (int i=0; i<4; i++) {
////                if (shank_Pos_difference[i] < 1e-2)
////                    return true;
////            }
//        }
        terminalReward = 0.f;

        return false;
    }

    private:
    int gcDim_, gvDim_, nJoints_;
    bool visualizable_ = false;
    raisim::ArticulatedSystem *anymal_;
    raisim::Box *anymal_box_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
    double terminalRewardCoeff_ = -10.;
    Eigen::VectorXd actionMean_, actionStd_, obDouble_;
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
    std::set<size_t> footIndices_;

    Eigen::VectorXd torque, reward_log, previous_action, target_postion, before_user_command, after_user_command;
    Eigen::Vector3d roll_and_pitch;
    Eigen::Vector4d foot_idx, shank_idx;
    size_t numContact_;
    size_t numFootContact_;
    std::vector<raisim::Vec<3>> footPos_;
    std::vector<raisim::Vec<3>> footPos_W;
    std::vector<raisim::Vec<3>> footVel_W;
    std::vector<Eigen::Vector3d> footContactVel_;
    std::array<bool, 4> footContactState_;
    Eigen::Vector4d foot_Pos_difference, shank_Pos_difference;
    int n_history_steps = 2;
    Eigen::VectorXd joint_position_error_history, joint_velocity_history, GRF_impulse;

    /// Lidar
    int scanSize;
    double lidar_theta, delta_lidar_theta;
    std::vector<raisim::Visuals *> scans;
    Eigen::VectorXd lidar_scan_depth;

    /// Random data generator
    int random_seed = 0;

    /// Randomization
    bool randomization = false, random_initialize = false, random_external_force = false;
    int random_external_force_final = 0, random_external_force_direction = 0;
    std::vector<raisim::Vec<3>> defaultJointPositions_;
    std::vector<double> defaultBodyMasses_;
    raisim::Vec<3> COMPosition_;

    /// Randon intialization & Random external force
    Eigen::VectorXd random_gc_init, random_gv_init, current_random_gc_init, current_random_gv_init;
    int random_init_n_step = 0, random_force_n_step = 0, random_force_period = 100, current_n_step = 0;

    /// Heightmap
    double hm_sizeX, hm_sizeY;
    raisim::HeightMap* hm;

    /// Traning configuration
    int total_traj_len, command_len;

    /// Obstacle
    int n_obstacle = 0;
    double obstacle_dr = 0.5;
    std::vector<double> hm_raw_value = {};
    std::vector<raisim::Vec<2>> init_set = {};
    int n_init_set = 0;
    Eigen::MatrixXd obstacle_centers;

    /// Reward (Cost) - New
    double obstacle_distance_cost = 0.0, command_similarity_cost = 0.0;
    double reward_obstacle_distance_coeff, reward_command_similarity_coeff;

    /// Observation to be predicted
    Eigen::VectorXd coordinateDouble;

    /// Trajectory prediction
    int n_prediction_step = 12;   // Determined manually
    std::vector<raisim::Visuals *> desired_command_traj, modified_command_traj;

    /// world type
    int world_type;
    int num_set_goal=0;

    std::vector<raisim::Visuals *> prediction_box_1, prediction_box_2;

    };
}
