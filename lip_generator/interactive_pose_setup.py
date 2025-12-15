"""
Interactive Robot Pose Setup with Visualization

This script allows you to:
1. Load and visualize the robot in meshcat
2. Manually adjust joint angles
3. See the robot update in real-time
4. Export the final pose configuration
"""

import numpy as np
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import time
import os


def load_robot():
    """Load the robot model with meshcat visualization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "model", "T1_7dof_arms_with_gripper.urdf")
    package_dir = os.path.join(script_dir, "model")

    robot = pinocchio.RobotWrapper.BuildFromURDF(
        urdf_path,
        package_dirs=[package_dir],
        root_joint=pinocchio.JointModelFreeFlyer(),
    )

    # Define default half_sitting pose
    half_sitting = np.array([
        0, 0, 0.665,  # base position (x, y, z)
        0, 0, 0, 1,   # base orientation (qx, qy, qz, qw)
        0, 0,         # torso joints
        0.2, -1.35, 0, -0.5, 0.0, 0.0, 0.0,  # left arm (7 joints)
        0.2, 1.35, 0, 0.5, 0.0, 0.0, 0.0,    # right arm (7 joints)
        0,            # head
        -0.2, 0.007658, 0, 0.4, -0.25, 0,    # left leg (6 joints)
        -0.2, -0.007658, 0, 0.4, -0.25, 0,   # right leg (6 joints)
    ])
    robot.model.referenceConfigurations["half_sitting"] = half_sitting

    return robot


def setup_visualizer(robot):
    """Set up meshcat visualizer."""
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install Python meshcat")
        print(err)
        return None

    try:
        viz.loadViewerModel()
        print("\n" + "=" * 80)
        print("Meshcat Visualizer Started!")
        print("=" * 80)
        print("Open your browser at: http://127.0.0.1:7000/static/")
        print("=" * 80 + "\n")
    except AttributeError as err:
        print("Error while loading the viewer model.")
        print(err)
        return None

    return viz


def print_joint_info(robot):
    """Print information about robot joints."""
    print("\n" + "=" * 80)
    print("ROBOT JOINT INFORMATION")
    print("=" * 80)
    print(f"Total configuration dimension (nq): {robot.model.nq}")
    print(f"Total velocity dimension (nv): {robot.model.nv}")
    print("\nJoint Structure:")
    print("-" * 80)

    joint_names = []
    for i, name in enumerate(robot.model.names[1:], 1):  # Skip universe
        joint = robot.model.joints[i]
        nq = joint.nq
        nv = joint.nv
        idx_q = joint.idx_q
        idx_v = joint.idx_v

        print(f"Joint {i}: {name}")
        print(f"  Type: {joint.shortname()}")
        print(f"  Config indices (q): {idx_q} to {idx_q + nq - 1 if nq > 0 else idx_q}")
        print(f"  Velocity indices (v): {idx_v} to {idx_v + nv - 1 if nv > 0 else idx_v}")
        joint_names.append(name)

    print("\n" + "=" * 80)
    print("CONFIGURATION VECTOR LAYOUT (36 elements)")
    print("=" * 80)
    print("Indices  0-2  : Base position (x, y, z)")
    print("Indices  3-6  : Base orientation (qx, qy, qz, qw)")
    print("Indices  7-8  : Torso (2 joints)")
    print("Indices  9-15 : Left arm (7 joints)")
    print("Indices 16-22 : Right arm (7 joints)")
    print("Index   23    : Head (1 joint)")
    print("Indices 24-29 : Left leg (6 joints)")
    print("Indices 30-35 : Right leg (6 joints)")
    print("=" * 80 + "\n")


def display_current_pose(robot, viz, q, name="Current Pose"):
    """Display robot in given configuration."""
    viz.display(q)

    # Compute and display foot positions
    pinocchio.forwardKinematics(robot.model, robot.data, q)
    pinocchio.updateFramePlacements(robot.model, robot.data)

    lf_id = robot.model.getFrameId("left_foot_link")
    rf_id = robot.model.getFrameId("right_foot_link")

    lf_pos = robot.data.oMf[lf_id].translation
    rf_pos = robot.data.oMf[rf_id].translation
    com_pos = pinocchio.centerOfMass(robot.model, robot.data, q)

    print(f"\n{name}:")
    print(f"  Base height: {q[2]:.4f} m")
    print(f"  Left foot:   [{lf_pos[0]:7.4f}, {lf_pos[1]:7.4f}, {lf_pos[2]:7.4f}]")
    print(f"  Right foot:  [{rf_pos[0]:7.4f}, {rf_pos[1]:7.4f}, {rf_pos[2]:7.4f}]")
    print(f"  CoM:         [{com_pos[0]:7.4f}, {com_pos[1]:7.4f}, {com_pos[2]:7.4f}]")


def interactive_mode(robot, viz):
    """Interactive mode to manually adjust pose."""
    print("\n" + "=" * 80)
    print("INTERACTIVE POSE ADJUSTMENT MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  'show'        - Display current configuration")
    print("  'reset'       - Reset to half_sitting pose")
    print("  'set <i> <v>' - Set joint i to value v (e.g., 'set 9 0.5')")
    print("  'base <x y z>'- Set base position (e.g., 'base 0 0 0.7')")
    print("  'export'      - Export current configuration to file")
    print("  'help'        - Show this help message")
    print("  'quit'        - Exit interactive mode")
    print("=" * 80)

    # Start with half_sitting
    q = robot.model.referenceConfigurations["half_sitting"].copy()
    display_current_pose(robot, viz, q, "Initial Pose (half_sitting)")

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                print("Exiting interactive mode...")
                break

            elif cmd == 'show' or cmd == 's':
                display_current_pose(robot, viz, q)
                print(f"\nFull configuration vector:")
                print(q)

            elif cmd == 'reset' or cmd == 'r':
                q = robot.model.referenceConfigurations["half_sitting"].copy()
                display_current_pose(robot, viz, q, "Reset to half_sitting")

            elif cmd.startswith('set '):
                parts = cmd.split()
                if len(parts) == 3:
                    idx = int(parts[1])
                    value = float(parts[2])
                    if 0 <= idx < len(q):
                        q[idx] = value
                        display_current_pose(robot, viz, q, f"Updated q[{idx}]={value}")
                    else:
                        print(f"Error: Index {idx} out of range [0, {len(q)-1}]")
                else:
                    print("Error: Usage: set <index> <value>")

            elif cmd.startswith('base '):
                parts = cmd.split()
                if len(parts) == 4:
                    q[0] = float(parts[1])  # x
                    q[1] = float(parts[2])  # y
                    q[2] = float(parts[3])  # z
                    display_current_pose(robot, viz, q, "Updated base position")
                else:
                    print("Error: Usage: base <x> <y> <z>")

            elif cmd == 'export' or cmd == 'e':
                filename = f"custom_pose_{int(time.time())}.npy"
                np.save(filename, q)
                print(f"\nConfiguration exported to: {filename}")
                print(f"Load with: q = np.load('{filename}')")

                # Also print as Python code
                print(f"\nPython code:")
                print("q = np.array([")
                for i in range(0, len(q), 6):
                    vals = ", ".join(f"{v:8.5f}" for v in q[i:min(i+6, len(q))])
                    print(f"    {vals},")
                print("])")

            elif cmd == 'help' or cmd == 'h' or cmd == '?':
                print("\nCommands:")
                print("  'show'        - Display current configuration")
                print("  'reset'       - Reset to half_sitting pose")
                print("  'set <i> <v>' - Set joint i to value v")
                print("  'base <x y z>'- Set base position")
                print("  'export'      - Export current configuration")
                print("  'help'        - Show this help message")
                print("  'quit'        - Exit")

            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for commands.")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def preset_poses(robot, viz):
    """Show some preset poses as examples."""
    print("\n" + "=" * 80)
    print("PRESET POSES")
    print("=" * 80)

    # Pose 1: Half sitting (default)
    q1 = robot.model.referenceConfigurations["half_sitting"].copy()
    display_current_pose(robot, viz, q1, "Preset 1: Half Sitting")
    time.sleep(2)

    # Pose 2: Wider stance
    q2 = q1.copy()
    q2[2] = 0.65  # Lower base slightly
    q2[24] = -0.25  # Left hip yaw
    q2[30] = -0.25  # Right hip yaw
    display_current_pose(robot, viz, q2, "Preset 2: Wider Stance")
    time.sleep(2)

    # Pose 3: Arms raised
    q3 = q1.copy()
    q3[9] = 1.5    # Left shoulder pitch
    q3[16] = 1.5   # Right shoulder pitch
    display_current_pose(robot, viz, q3, "Preset 3: Arms Raised")
    time.sleep(2)

    # Back to default
    display_current_pose(robot, viz, q1, "Back to Half Sitting")

    return q1


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("INTERACTIVE ROBOT POSE SETUP")
    print("=" * 80)

    # Load robot
    print("\nLoading robot model...")
    robot = load_robot()
    print("✓ Robot loaded successfully")

    # Print joint info
    print_joint_info(robot)

    # Setup visualizer
    print("\nSetting up visualizer...")
    viz = setup_visualizer(robot)

    if viz is None:
        print("Failed to initialize visualizer. Exiting.")
        return

    print("✓ Visualizer ready")

    # Show preset poses
    input("\nPress Enter to see preset poses...")
    q_default = preset_poses(robot, viz)

    # Interactive mode
    input("\nPress Enter to start interactive mode...")
    interactive_mode(robot, viz)

    print("\n" + "=" * 80)
    print("Session complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
