//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <time.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

// [Tip]
//
// // Logging example
// Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
// std::cout << current_leg_phase.format(CommaInitFmt) << std::endl;
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

            world_->addGround();
            random_seed = seed;

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_ + "/anymal_c/urdf/anymal.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim(); // 3(base position) + 4(base orientation) + 12(joint position) = 19
            gvDim_ = anymal_->getDOF();                      // 3(base linear velocity) + 3(base angular velocity) + 12(joint velocity) = 18
            nJoints_ = gvDim_ - 6;                           // 12

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
            user_command.setZero(3);

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
            reward_joint_torque_coeff = cfg["reward"]["joint_torque"]["coeff"].As<double>();
            reward_linear_vel_coeff = cfg["reward"]["linear_vel_error"]["coeff"].As<double>();
            reward_angular_vel_coeff = cfg["reward"]["angular_vel_error"]["coeff"].As<double>();
            reward_joint_vel_coeff = cfg["reward"]["joint_vel"]["coeff"].As<double>();
            reward_foot_clearance_coeff = cfg["reward"]["foot_clearance"]["coeff"].As<double>();
            reward_foot_slip_coeff = cfg["reward"]["foot_slip"]["coeff"].As<double>();
            reward_previous_action_smooth_coeff = cfg["reward"]["previous_action_smooth"]["coeff"].As<double>();
            reward_foot_z_vel_coeff = cfg["reward"]["foot_z_vel"]["coeff"].As<double>();
            reward_orientation_coeff = cfg["reward"]["orientation"]["coeff"].As<double>();

//            /// user commmand range
//            min_forward_vel = cfg["command"]["forward_vel"]["min"].As<double>();
//            max_forward_vel = cfg["command"]["forward_vel"]["max"].As<double>();
//            min_lateral_vel = cfg["command"]["lateral_vel"]["min"].As<double>();
//            max_lateral_vel = cfg["command"]["lateral_vel"]["max"].As<double>();
//            min_yaw_rate = cfg["command"]["yaw_rate"]["min"].As<double>();
//            max_yaw_rate = cfg["command"]["yaw_rate"]["max"].As<double>();

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
            gc_init_ << 0, 0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.03, 0.5, -0.9, -0.03, 0.5, -0.9, 0.03, -0.5, 0.9, -0.03, -0.5, 0.9;  //0.5
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
            obDim_ = 81;
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

            /// visualize if it is the first environment
            if (visualizable_)
            {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();

                server_->focusOn(anymal_);
            }
        }

    void init() final {}

    void reset() final
    {
        static std::default_random_engine generator(random_seed);

        if (random_initialize) {
            if (current_n_step == 0) {
                raisim::Vec<3> random_axis;
                raisim::Vec<4> random_quaternion;

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
//        clock_t start, end;
//        double result;
//
//        start = clock();
//        current_n_step += 1;

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
                if (random_force_n_step <= current_n_step && current_n_step < random_force_n_step + random_force_period) {
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

//        end = clock();
//        result = (double) (end - start);
//        std::cout << "result (PD) : " << ((result) / CLOCKS_PER_SEC) << " seconds" << std::endl;


//        start = clock();
        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
        {
            if (server_)
                server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_)
                server_->unlockVisualizationServerMutex();
        }
//        end = clock();
//        result = (double) (end - start);
//        std::cout << "result (integrate) : " << ((result) / CLOCKS_PER_SEC) << " seconds" << std::endl;

//        start = clock();
        updateObservation();
//        end = clock();
//        result = (double) (end - start);
//        std::cout << "result (observe) : " << ((result) / CLOCKS_PER_SEC) << " seconds" << std::endl;

        return 0.0;

//        torque = anymal_->getGeneralizedForce().e(); // squaredNorm

//        calculate_cost();

//        rewards_.record("joint_torque", -torqueCost);
//        rewards_.record("linear_vel_error", -linvelCost);
//        rewards_.record("angular_vel_error", -angVelCost);
//        rewards_.record("joint_vel", -velLimitCost);
//        rewards_.record("foot_clearance", -footClearanceCost);
//        rewards_.record("foot_slip", -slipCost);
//        rewards_.record("previous_action_smooth", -previousActionCost);
//        rewards_.record("foot_z_vel", -footVelCost);
//        rewards_.record("orientation", -orientationCost);
//
//        previous_action = target_postion.cast<double>();

//        return rewards_.sum();
    }

    void updateObservation() {
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

        obDouble_ << rot.e().row(2).transpose(),      /// body orientation (dim=3)
                     gc_.tail(12),                    /// joint angles (dim=12)
                     bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity (dim=3+3=6)
                     gv_.tail(12),                  /// joint velocity (dim=12)
                     joint_position_error_history,    /// joint position error history (dim=24)
                     joint_velocity_history;          /// joint velocity history (dim=24)

        /// Update coordinate
        double yaw = atan2(rot.e().col(0)[1], rot.e().col(0)[0]);

        coordinateDouble << gc_[0], gc_[1], yaw;

//        roll_and_pitch = rot.e().row(2).transpose();
//        GRF_impulse.setZero(4);

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
        ob = obDouble_.cast<float>();
    }

    void coordinate_observe(Eigen::Ref<EigenVec> coordinate)
    {
        /// convert it to float
        /// 0 : x coordinate
        /// 1 : y coordinate
        /// 2 : yaw angle
        coordinate = coordinateDouble.cast<float>();
    }

    void calculate_cost()
    {
        torqueCost=0, linvelCost=0, angVelCost=0, velLimitCost = 0, footClearanceCost = 0, slipCost = 0;
        previousActionCost = 0, footVelCost = 0, orientationCost = 0;
    }

    void comprehend_contacts()
    {
        numContact_ = anymal_->getContacts().size();

        // numFootContact_ = 0;
        Eigen::Vector4d foot_Pos_height_map;
        float shank_dr = 0.06;

        for (int k = 0; k < 4; k++)
        {
            footContactState_[k] = false;
            anymal_->getFramePosition(foot_idx[k], footPos_W[k]);  //position of the feet
            anymal_->getFrameVelocity(foot_idx[k], footVel_W[k]);
//            foot_Pos_height_map[k] = hm->getHeight(footPos_W[k][0], footPos_W[k][1]);
            foot_Pos_height_map[k] = 0.;   /// Should change if it is rough terrain!!!!
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
                    if (idx == 3 && foot_Pos_difference[0] < dr && !footContactState_[0])
                    {
                        footContactState_[0] = true;
                        // footNormal_[0] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[0] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[0] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 6 && foot_Pos_difference[1] < dr && !footContactState_[1])
                    {
                        footContactState_[1] = true;
                        // footNormal_[1] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[1] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[1] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 9 && foot_Pos_difference[2] < dr && !footContactState_[2])
                    {
                        footContactState_[2] = true;
                        // footNormal_[2] = anymal_->getContacts()[k].getNormal().e();
                        anymal_->getContactPointVel(k, vec3);
                        footContactVel_[2] = vec3.e();
                        // numFootContact_++;
                        GRF_impulse[2] = anymal_->getContacts()[k].getImpulse().e().squaredNorm();
                    }
                    else if (idx == 12 && foot_Pos_difference[3] < dr && !footContactState_[3])
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
                                        double collision_threshold) {}

    void visualize_modified_command_traj(Eigen::Ref<EigenRowMajorMat> coordinate_modified_command,
                                         Eigen::Ref<EigenVec> P_col_modified_command,
                                         double collision_threshold) {}

    void set_user_command(Eigen::Ref<EigenVec> command) {}

    void reward_logging(Eigen::Ref<EigenVec> rewards) {}

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

    void set_goal(Eigen::Ref<EigenVec> goal_pos) {}

    void baseline_compute_reward(Eigen::Ref<EigenRowMajorMat> sampled_command,
                                 Eigen::Ref<EigenVec> goal_Pos_local,
                                 Eigen::Ref<EigenVec> rewards_p,
                                 Eigen::Ref<EigenVec> collision_idx,
                                 int steps, double delta_t, double must_safe_time) {}

    void initialize_n_step()
    {
        current_n_step = 0;
    }

    void computed_heading_direction(Eigen::Ref<EigenVec> heading_direction_) {}

    bool isTerminalState(float &terminalReward) final
    {
        terminalReward = float(terminalRewardCoeff_);

        /// if the contact body is not feet
        for (auto &contact : anymal_->getContacts()) {
            if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
                return true;
            }
//            for (int i=0; i<4; i++) {
//                if (shank_Pos_difference[i] < 1e-2)
//                    return true;
//            }
        }
        terminalReward = 0.f;

        /// if the robot is out of the terrain
        if (fabs(gc_[0])>25. || fabs(gc_[1])>25.)
            return true;

        return false;
    }

    void curriculumUpdate() final {}

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -10.;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;

        Eigen::VectorXd torque, reward_log, previous_action, target_postion, user_command;
        Eigen::Vector3d roll_and_pitch;
        Eigen::Vector4d foot_idx, shank_idx;
        size_t numContact_;
        size_t numFootContact_;
        std::vector<raisim::Vec<3>> footPos_;
        std::vector<raisim::Vec<3>> footPos_W;
        std::vector<raisim::Vec<3>> footVel_W;
        std::vector<Eigen::Vector3d> footContactVel_;
        std::array<bool, 4> footContactState_;
        double costScale_ = 0.3, costScale2_ = 0.3;
        double torqueCost=0, linvelCost=0, angVelCost=0, velLimitCost = 0, footClearanceCost = 0, slipCost = 0;
        double previousActionCost = 0, footVelCost = 0, orientationCost = 0, cost;
        double reward_joint_torque_coeff, reward_linear_vel_coeff, reward_angular_vel_coeff, reward_joint_vel_coeff, reward_foot_clearance_coeff;
        double reward_foot_slip_coeff, reward_previous_action_smooth_coeff, reward_foot_z_vel_coeff, reward_orientation_coeff;
        int yaw_scanSize, pitch_scanSize;
        raisim::HeightMap* hm;
        Eigen::Vector4d foot_Pos_difference, shank_Pos_difference;
        int n_history_steps = 2;
        Eigen::VectorXd joint_position_error_history, joint_velocity_history, GRF_impulse;

        /// Randomization
        bool randomization = false, random_initialize = false, random_external_force = false;
        int random_external_force_final = 0, random_external_force_direction = 0;
        std::vector<raisim::Vec<3>> defaultJointPositions_;
        std::vector<double> defaultBodyMasses_;
        raisim::Vec<3> COMPosition_;

        /// Randon intialization & Random external force
        Eigen::VectorXd random_gc_init, random_gv_init, current_random_gc_init, current_random_gv_init;
        int random_init_n_step = 0, random_force_n_step = 0, random_force_period = 100, current_n_step = 0;

        /// User command
        double min_forward_vel, max_forward_vel, min_lateral_vel, max_lateral_vel, min_yaw_rate, max_yaw_rate;
        int total_traj_len, command_len;

        /// Seed
        int random_seed;

        /// Observation to be predicted
        Eigen::VectorXd coordinateDouble;

    };
}
