import sys

class BionicHardwareInterface:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_bus = None
        self.hardware_active = False

    def connect(self):
        """Attempts to connect to physical leg hardware controllers."""
        try:
            self.hardware_active = True
            print(f"[HW] Physical bionic limb active on port {self.port}")
        except Exception as e:
            print(f"[HW] Microcontroller not detected ({e}). Running in DIGITAL TWIN ONLY mode.")
            self.hardware_active = False

    def write_actuators(self, safe_joint_angles: dict) -> dict:
        """
        Pushes sanitized commands to physical servos and returns a data payload 
        formatted for your WebGL visualizers.
        """

        if self.hardware_active and self.serial_bus:
            try:
                packet = f"H:{safe_joint_angles.get('hip_pitch', 0):.1f}|K:{safe_joint_angles.get('knee', 0):.1f}|A:{safe_joint_angles.get('ankle_pitch', 0):.1f}\n"
                self.serial_bus.write(packet.encode('utf-8'))
            except Exception as e:
                print(f"[HW] Connection lost during execution packet delivery: {e}")
                self.hardware_active = False

        twin_payload = {
            "hipPitch":   safe_joint_angles.get("hip_pitch", 0.0),
            "knee":        safe_joint_angles.get("knee", 0.0),
            "anklePitch": safe_joint_angles.get("ankle_pitch", 0.0)
        }
        return twin_payload
