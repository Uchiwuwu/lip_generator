import numpy as np
import pinocchio

import crocoddyl


class SimpleBipedStairProblem:
    """Build simple bipedal stair locomotion problems.

    This class provides functionality for creating optimal control problems
    for bipedal robots walking on stairs.
    """

    def __init__(
        self,
        rmodel,
        rightFoot,
        leftFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
    ):
        """Construct biped-stair problem.

        :param rmodel: robot model
        :param rightFoot: name of the right foot
        :param leftFoot: name of the left foot
        :param integrator: type of the integrator
            (options are: 'euler', and 'rk4')
        :param control: type of control parametrization
            (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate(
            [q0, np.zeros(self.rmodel.nv)])
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

    def createSingleStairStepProblem(
        self, x0, leftFootTarget, rightFootTarget, timeStep, stepKnots, supportKnots, stepHeight=0.10, targetYaw=0.0, transitionKnots=None, comShiftRatio=0.8, initialComShift=0.8
    ):
        """Create a shooting problem for a single step with specified foot locations.

        :param x0: initial state
        :param leftFootTarget: target position (3D) for left foot [x, y, z]
        :param rightFootTarget: target position (3D) for right foot [x, y, z]
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :param stepHeight: height of foot swing trajectory (default: 0.10m)
        :param targetYaw: target yaw angle for swing foot (default: 0.0 rad)
        :param transitionKnots: number of knots for post-swing COM centering phase (default: same as supportKnots)
        :param comShiftRatio: ratio of COM shift towards center during swing (default: 0.8 = 80%)
        :param initialComShift: ratio of COM shift towards stance foot in initial phase (default: 0.8 = 80%)
        :return shooting problem
        """
        # Default transitionKnots to supportKnots if not specified
        if transitionKnots is None:
            transitionKnots = supportKnots
        # Compute the current foot positions and orientations
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        # Get initial yaw angles of both feet in world frame
        rfRot0 = self.rdata.oMf[self.rfId].rotation
        lfRot0 = self.rdata.oMf[self.lfId].rotation
        # Extract yaw from rotation matrix (yaw = atan2(R[1,0], R[0,0]))
        rfYaw0 = np.arctan2(rfRot0[1, 0], rfRot0[0, 0])
        lfYaw0 = np.arctan2(lfRot0[1, 0], lfRot0[0, 0])

        # Get initial base yaw from base orientation
        # Base orientation is stored as quaternion [x, y, z, w] in q0[3:7]
        base_quat = pinocchio.Quaternion(q0[6], q0[3], q0[4], q0[5])  # [w, x, y, z]
        base_rot = base_quat.toRotationMatrix()
        baseYaw0 = np.arctan2(base_rot[1, 0], base_rot[0, 0])

        # Compute CoM reference between current foot positions
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        # Determine which foot needs to move
        leftFootTarget = np.array(leftFootTarget)
        rightFootTarget = np.array(rightFootTarget)

        leftFootMovement = np.linalg.norm(leftFootTarget - lfPos0)
        rightFootMovement = np.linalg.norm(rightFootTarget - rfPos0)

        loco3dModel = []
        if leftFootMovement > rightFootMovement:
            stance_foot_pos = rfPos0
        else:
            stance_foot_pos = lfPos0

        # Calculate COM shifted toward stance foot
        # Use actual current COM height (not lowered comRef) to prevent dropping
        com_current_height = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]
        com_displacement_initial = stance_foot_pos[:2] - comRef[:2]
        com_initial_shifted = comRef[:2] + com_displacement_initial * initialComShift
        com_initial_shifted_3d = np.array([com_initial_shifted[0], com_initial_shifted[1], com_current_height])

        # Initial double support phase - shift COM toward stance foot to prepare for swing
        doubleSupport_initial = [
            self.createSwingFootModel(
                timeStep,
                [self.rfId, self.lfId],
                comTask=com_initial_shifted_3d,
                comWeight=5e8,
            )
            for k in range(supportKnots)
        ]
        loco3dModel += doubleSupport_initial

        # Calculate final COM position (center between both feet)
        com_final = (leftFootTarget + rightFootTarget) / 2
        com_final[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        # Determine which foot to move first (the one with larger movement)
        if leftFootMovement > rightFootMovement:
            # Move left foot first (right foot is stance)
            lStep = self.createFootStairstepModelsWithTarget(
                comRef,
                lfPos0,
                leftFootTarget,
                stepHeight,
                timeStep,
                stepKnots,
                [self.rfId],  # right foot supports
                [self.lfId],  # left foot swings
                targetYaw,
                comShiftRatio,
            )
            loco3dModel += lStep

            # Add transition double support - move COM to center between both feet
            doubleSupport_transition = [
                self.createSwingFootModel(
                    timeStep,
                    [self.rfId, self.lfId],
                    comTask=com_final,
                    comWeight=5e8,
                )
                for k in range(transitionKnots)
            ]
            loco3dModel += doubleSupport_transition
        else:
            # Move right foot first (left foot is stance)
            rStep = self.createFootStairstepModelsWithTarget(
                comRef,
                rfPos0,
                rightFootTarget,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfId],  # left foot supports
                [self.rfId],  # right foot swings
                targetYaw,
                comShiftRatio,
            )
            loco3dModel += rStep

            # Add transition double support - move COM to center between both feet
            doubleSupport_transition = [
                self.createSwingFootModel(
                    timeStep,
                    [self.rfId, self.lfId],
                    comTask=com_final,
                    comWeight=5e8,
                )
                for k in range(transitionKnots)
            ]
            loco3dModel += doubleSupport_transition

        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
    def createFootStairstepModelsWithTarget(
        self,
        comPos0,
        footPos0,
        footTarget,
        stepHeight,
        timeStep,
        numKnots,
        supportFootIds,
        swingFootIds,
        targetYaw=0.0,
        comShiftRatio=0.8,
        baseYawDuringSwing=None,
    ):
        """Action models for a footstep phase with explicit target position.

        :param comPos0: initial CoM position
        :param footPos0: initial position of the swinging foot
        :param footTarget: target position (3D) for the swinging foot
        :param stepHeight: step height for swing trajectory
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :param targetYaw: target yaw angle for the swinging foot (default: 0.0 rad)
        :param comShiftRatio: ratio of COM shift towards center during swing (default: 0.8 = 80%)
        :param baseYawDuringSwing: yaw angle to maintain for the base during swing (default: None)
        :return footstep action models
        """
        # Convert to numpy arrays
        footPos0 = np.array(footPos0)
        footTarget = np.array(footTarget)

        # Compute total displacement
        displacement = footTarget - footPos0

        # Separate horizontal (xy) and vertical (z) displacements
        displacement_xy = displacement[:2]  # x and y components
        displacement_z = displacement[2]     # z component (stair height)

        # Get stance foot position (for COM calculation)
        # Assume the stance foot position is available from the robot data
        stance_foot_pos = self.rdata.oMf[supportFootIds[0]].translation.copy()

        # Calculate target center of support (between stance foot and landing position)
        target_center = (stance_foot_pos + footTarget) / 2
        target_center[2] = comPos0[2]  # Keep same height

        # COM should shift towards target center based on the specified ratio
        com_shift_amount = comShiftRatio
        
        if displacement_z < 0:
            stepupHeight = stepHeight - np.abs(displacement_z)
            stepdownHeight = stepHeight
        elif displacement_z > 0:
            stepdownHeight = stepHeight - np.abs(displacement_z)
            stepupHeight = stepHeight
        else:
            stepdownHeight = stepHeight
            stepupHeight = stepHeight
        # print(displacement_xy, displacement_z, stepupHeight, stepdownHeight)
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i in swingFootIds:
                # Create smooth trajectory from footPos0 to footTarget
                # Phase 1 (first half): swing up
                # Phase 2 (second half): swing down
                progress = (k + 1) / numKnots  # Linear progress from 0 to 1

                # UPSIDE-DOWN PARABOLA with flat peak
                xy_progress = progress

                # Create trajectory with 3 phases: swing up, plateau, swing down
                plateau_fraction = 0.2  # Middle 20% is flat at the peak
                swing_up_end = 0.5 - plateau_fraction / 2  # 0.4
                swing_down_start = 0.5 + plateau_fraction / 2  # 0.6
                if progress < swing_up_end:
                    # Swing up phase - use quintic polynomial
                    t = progress / swing_up_end  # Normalize to 0-1
                    z_height = stepupHeight * (10 * t**3 - 15 * t**4 + 6 * t**5)
                elif progress < swing_down_start:
                    # Plateau phase - stay at peak
                    z_height = stepupHeight
                else:
                    # Swing down phase - use quintic polynomial
                    t = (progress - swing_down_start) / (1.0 - swing_down_start)  # Normalize to 0-1
                    z_height = stepupHeight - stepdownHeight * (10 * t**3 - 15 * t**4 + 6 * t**5)

                # Interpolate x,y position (horizontal only)
                tref = np.zeros(3)
                tref[:2] = footPos0[:2] + displacement_xy * xy_progress

                # For stairs: base height interpolates from start to target
                tref[2] = footPos0[2] + z_height
                # print(tref[2])
                # Create rotation matrix for target yaw (rotation around z-axis)
                # Interpolate yaw from 0 to targetYaw
                current_yaw = targetYaw * progress
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                R_target = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw,  cos_yaw, 0],
                    [0,        0,       1]
                ])

                swingFootTask += [[i, pinocchio.SE3(R_target, tref)]]

            # Update CoM task - shift 80% towards center of stance and landing feet
            # comPos0 is initial COM, target_center is the center between stance and landing
            com_displacement = target_center[:2] - comPos0[:2]
            comTask_xy = comPos0[:2] + com_displacement * progress * com_shift_amount

            # For stairs: adjust COM height based on stair direction
            if displacement_z < 0:  # Descending
                # COM shifts down by 0.1m during descent
                comTask_z = comPos0[2] - 0.1 * progress
            else:  # Ascending or level
                # COM stays at initial height
                comTask_z = comPos0[2]

            comTask = np.array([comTask_xy[0], comTask_xy[1], comTask_z])

            footSwingModel += [
                self.createSwingFootModel(
                    timeStep,
                    supportFootIds,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                    footWeight=1e9,  # Very high weight to enforce straight line trajectory in x,y
                    baseYawTask=baseYawDuringSwing,  # Keep base yaw fixed during swing
                    # baseYawWeight=1e10,  # Moderate weight (hip yaw constraints do most of the work)
                    # progressRatio=progress,  # Pass current progress for landing damping
                    # landingDampingStart=0.5,  # Start damping at 70% of swing
                    # landingDampingWeight=5e3  # Base weight for velocity penalty
                )
            ]

        # Action model for the foot switch (landing)
        footSwitchModel = self.createFootSwitchModel(
            swingFootIds, swingFootTask, baseYawTask=baseYawDuringSwing
        )

        return [*footSwingModel, footSwitchModel]
    def createSwingFootModel(
        self, timeStep, supportFootIds, comTask=None, swingFootTask=None, comWeight=1e5, footWeight=1e6,
        progressRatio=None, landingDampingStart=0.7, landingDampingWeight=1e5, baseYawTask=None, baseYawWeight=1e6
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :param comWeight: weight for COM tracking (default: 1e5, higher for double support)
        :param footWeight: weight for swing foot position tracking (default: 1e6, increase to improve reaching)
        :param progressRatio: current progress in swing (0-1), used for landing damping
        :param landingDampingStart: when to start landing damping (0-1, default 0.7 = 70%)
        :param landingDampingWeight: weight for velocity penalty during landing phase (default: 1e5)
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(
                self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, comWeight)

        state_target = self.rmodel.defaultState.copy()
        # Add base yaw orientation task if specified
        if baseYawTask is not None:
            # Create a pure yaw rotation (pitch=0, roll=0)
            # Convert yaw to quaternion: q = [w, x, y, z] where for yaw: w=cos(yaw/2), z=sin(yaw/2), x=y=0
            target_quat = pinocchio.Quaternion(
                np.cos(baseYawTask / 2),  # w
                0.0,  # x
                0.0,  # y
                np.sin(baseYawTask / 2)  # z
            )
            target_quat.normalize()

            # Create modified state target
            # Set base orientation (indices 3-6 in configuration)
            state_target[3] = target_quat.x  # qx
            state_target[4] = target_quat.y  # qy
            state_target[5] = target_quat.z  # qz
            state_target[6] = target_quat.w  # qw

            # Create state residual with custom activation for base orientation only
            baseOrientationResidual = crocoddyl.ResidualModelState(
                self.state, state_target, nu
            )
            # Weight only the base orientation (indices 3-5 in velocity space)
            orientation_weights = np.zeros(self.state.ndx)
            orientation_weights[3:6] = 1.0  # Base orientation components
            baseOrientationActivation = crocoddyl.ActivationModelWeightedQuad(
                orientation_weights**2
            )
            baseOrientationCost = crocoddyl.CostModelResidual(
                self.state, baseOrientationActivation, baseOrientationResidual
            )
            costModel.addCost("baseYawTrack", baseOrientationCost, baseYawWeight)  # Use customizable weight

        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(
                self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, footWeight
                )

                # PRE-LANDING VELOCITY DAMPING - Z-AXIS ONLY
                # Apply stronger velocity penalty during landing phase for smoother deceleration
                if progressRatio is not None and progressRatio > landingDampingStart:
                    # Increase weight during landing phase (final 30% of swing)
                    landing_phase = (progressRatio - landingDampingStart) / (1.0 - landingDampingStart)
                    # Quadratic increase in damping: starts at base, peaks at landing
                    adaptive_damping_weight = landingDampingWeight * (landing_phase ** 2)

                    frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                        self.state, i[0], pinocchio.Motion.Zero(),
                        pinocchio.LOCAL_WORLD_ALIGNED, nu
                    )

                    # Weight only the z-axis (index 2) for linear velocity
                    # Motion has 6 components: [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
                    velocity_weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    velocityActivation = crocoddyl.ActivationModelWeightedQuad(velocity_weights**2)

                    velocityDampingCost = crocoddyl.CostModelResidual(
                        self.state, velocityActivation, frameVelocityResidual
                    )
                    costModel.addCost(
                        self.rmodel.frames[i[0]].name + "_landingDamping_z",
                        velocityDampingCost,
                        adaptive_damping_weight
                    )

        # Add collision avoidance between swing and stance feet
        # This prevents feet from getting too close during the swing phase
        if len(supportFootIds) > 0 and swingFootTask is not None:
            for swing_task in swingFootTask:
                swing_foot_id = swing_task[0]
                for support_foot_id in supportFootIds:
                    # Skip if same foot
                    if swing_foot_id == support_foot_id:
                        continue

                    # Create a penalty for feet getting too close
                    # We penalize configurations where feet are closer than min_distance
                    try:
                        # Create frame velocity residual to indirectly discourage collision
                        # (Crocoddyl doesn't have direct distance constraints)
                        frame_residual = crocoddyl.ResidualModelFrameVelocity(
                            self.state, swing_foot_id, pinocchio.Motion.Zero(),
                            pinocchio.LOCAL, nu
                        )
                        # Weighted activation to smooth the cost
                        activation = crocoddyl.ActivationModelWeightedQuad(
                            np.array([1.0, 10.0, 1.0, 0.1, 0.1, 0.1]) ** 2
                        )
                        collision_cost = crocoddyl.CostModelResidual(
                            self.state, activation, frame_residual
                        )
                        costModel.addCost(
                            f"collision_avoid_{self.rmodel.frames[swing_foot_id].name}_{self.rmodel.frames[support_foot_id].name}",
                            collision_cost,
                            5e2  # Moderate weight
                        )
                    except:
                        pass  # Skip if not supported

        # Add dedicated base angular velocity cost to directly penalize angular velocity
        try:
            baseAngularVelResidual = crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self.rmodel.getFrameId("base_link"),
                pinocchio.Motion.Zero(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu
            )
            # Weight only the angular velocity (last 3 components of 6D motion)
            angularVelWeights = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            angularVelActivation = crocoddyl.ActivationModelWeightedQuad(angularVelWeights**2)
            baseAngularVelCost = crocoddyl.CostModelResidual(
                self.state, angularVelActivation, baseAngularVelResidual
            )
            costModel.addCost("baseAngularVel", baseAngularVelCost, 1e8)  # Original weight
        except:
            pass  # Skip if base_link frame doesn't exist

        # State weights with increased penalty for upper body joints
        # Upper body: torso(2) + left_arm(7) + right_arm(7) + head(1) = 17 joints at indices 6-22
        num_upper_body = 17
        num_leg_joints = self.state.nv - 6 - num_upper_body

        # Create leg joint weights with higher penalty for hip roll joints
        # Leg structure: [Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll] x 2 legs
        leg_joint_weights = []
        leg_joint_velocity_weights = []
        for leg in range(2):  # Left and right legs
            leg_joint_weights += [
                0.01,    # Hip_Pitch
                0.5,     # Hip_Roll - HIGHER weight to prevent excessive deviation
                0.1,     # Hip_Yaw - HIGH weight to prevent base rotation
                0.001,   # Knee_Pitch - LOWER weight to allow more bending
                0.01,    # Ankle_Pitch
                0.01,    # Ankle_Roll
            ]
            leg_joint_velocity_weights += [
                10,      # Hip_Pitch velocity
                10,      # Hip_Roll velocity
                1,     # Hip_Yaw velocity - VERY HIGH to prevent yaw rotation
                10,      # Knee_Pitch velocity
                10,      # Ankle_Pitch velocity
                10,      # Ankle_Roll velocity
            ]
        stateWeights = np.array(
            [0, 0, 0] +                          # base position (free)
            [5e3, 5e3, 100]  +                          # base orientation
            [10.0] * num_upper_body +           # upper body joints - HIGHER weight
            leg_joint_weights +                  # leg joints with higher hip roll weight
            [100, 100, 500] +                    # base linear velocity
            [5e3] * 3 +                          # base angular velocity - VERY HIGH
            [10] * num_upper_body +              # upper body joint velocities
            leg_joint_velocity_weights           # leg joint velocities
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, state_target, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e2)
        costModel.addCost("ctrlReg", ctrlReg, 1e3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        return model
    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False, baseYawTask=None):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :param baseYawTask: target yaw angle for the base (default: None)
        :return action model for a foot switch phase
        """
        return self.createImpulseModel(supportFootIds, swingFootTask, baseYawTask=baseYawTask)
    def createImpulseModel(
        self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0, baseYawTask=None
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param baseYawTask: target yaw angle for the base (default: None)
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, 1e8
                )
        # State weights with increased penalty for upper body joints
        num_upper_body = 17
        num_leg_joints = self.rmodel.nv - 6 - num_upper_body

        # Create leg joint weights with higher penalty for hip roll joints
        leg_joint_weights = []
        leg_joint_velocity_weights = []
        for leg in range(2):  # Left and right legs
            leg_joint_weights += [
                0.1,     # Hip_Pitch
                0.5,     # Hip_Roll - HIGHER weight to prevent excessive deviation
                0.01,    # Hip_Yaw
                0.01,    # Knee_Pitch - LOWER weight to allow more bending
                0.1,     # Ankle_Pitch
                0.1,     # Ankle_Roll
            ]
            leg_joint_velocity_weights += [
                10,      # Hip_Pitch velocity
                10,      # Hip_Roll velocity
                1,      # Hip_Yaw velocity
                10,      # Knee_Pitch velocity
                10,      # Ankle_Pitch velocity
                10,      # Ankle_Roll velocity
            ]

        # Create state target with custom base yaw if specified
        state_target = self.rmodel.defaultState.copy()
        if baseYawTask is not None:
            # Modify state target to have the desired base yaw
            target_quat = pinocchio.Quaternion(
                np.cos(baseYawTask / 2),  # w
                0.0,  # x
                0.0,  # y
                np.sin(baseYawTask / 2)  # z
            )
            target_quat.normalize()
            state_target[3] = target_quat.x  # qx
            state_target[4] = target_quat.y  # qy
            state_target[5] = target_quat.z  # qz
            state_target[6] = target_quat.w  # qw

        stateWeights = np.array(
            [0, 0, 0]+                         # base position
            [5e3, 5e3, 100] +                        # base orientation
            [10] * num_upper_body +           # upper body joints - HIGHER weight
            leg_joint_weights +                # leg joints with higher hip roll weight
            [100, 100, 500] +                  # base linear velocity
            [5e3] * 3 +                        # base angular velocity
            [10] * num_upper_body +            # upper body joint velocities
            leg_joint_velocity_weights         # leg joint velocities
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, state_target, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model
