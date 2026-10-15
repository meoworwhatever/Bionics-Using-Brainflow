class LegKinematicsEngine:
    def __init__(self):
        self.joint_profiles = {
            "hip_pitch":   {"neutral": 0.0,   "max_flex": 35.0},   # Degrees flexion
            "knee":        {"neutral": 0.0,   "max_flex": 65.0},   # Degrees flexion backward
            "ankle_pitch": {"neutral": 0.0,   "max_flex": -15.0}   # Plantarflexion balance shift
        }

    def process_intent_to_angles(self, neural_flex: float) -> dict:
        """
        Takes a single neural intent value (0.0 to 1.0) and maps it across a 
        coordinated joint movement synergy.
        """
        # Ensure input is clamped safely between 0.0 and 1.0
        flex_factor = max(0.0, min(1.0, neural_flex))
        
        target_angles = {}
        for joint, config in self.joint_profiles.items():
            neutral = config["neutral"]
            max_flex = config["max_flex"]
            
            # Linear scaling across the coordination matrix
            target_angles[joint] = neutral + (flex_factor * (max_flex - neutral))
            
        return target_angles

class LegKinematicsEngine:
    def __init__(self):
        self.joint_profiles = {
            "hip_pitch":   {"neutral": 0.0,   "max_flex": 35.0},   # Degrees flexion
            "knee":        {"neutral": 0.0,   "max_flex": 65.0},   # Degrees flexion backward
            "ankle_pitch": {"neutral": 0.0,   "max_flex": -15.0}   # Plantarflexion balance shift
        }

    def process_intent_to_angles(self, neural_flex: float) -> dict:
        """
        Takes a single neural intent value (0.0 to 1.0) and maps it across a 
        coordinated joint movement synergy.
        """
        # Ensure input is clamped safely between 0.0 and 1.0
        flex_factor = max(0.0, min(1.0, neural_flex))
        
        target_angles = {}
        for joint, config in self.joint_profiles.items():
            neutral = config["neutral"]
            max_flex = config["max_flex"]
            
            # Linear scaling across the coordination matrix
            target_angles[joint] = neutral + (flex_factor * (max_flex - neutral))
            
        return target_angles

class ArmKinematicsEngine:
    def __init__(self):
        # Maps neural intent to a fully articulated bionic upper limb
        self.joint_profiles = {
            "elbow_flex":     {"neutral": 0.0,   "max_flex": 130.0}, # Degrees elbow flexion
            "wrist_rotate":   {"neutral": 0.0,   "max_flex": 90.0},  # Pronation/Supination
            "wrist_pitch":    {"neutral": 0.0,   "max_flex": 45.0},  # Flexion/Extension
            "thumb_oppose":   {"neutral": 0.0,   "max_flex": 90.0},  # Opposition movement
            "index_flex":     {"neutral": 0.0,   "max_flex": 90.0},  # Finger closing synergy
            "middle_flex":    {"neutral": 0.0,   "max_flex": 90.0},  
            "ring_flex":      {"neutral": 0.0,   "max_flex": 90.0},  
            "pinky_flex":     {"neutral": 0.0,   "max_flex": 90.0}   
        }

    def process_intent_to_angles(self, neural_flex: float) -> dict:
        """
        Takes a single neural intent value (0.0 to 1.0) from BrainFlow 
        and maps it smoothly across the entire arm coordination matrix.
        """
        # Ensure input is clamped safely between 0.0 and 1.0
        flex_factor = max(0.0, min(1.0, neural_flex))
        
        target_angles = {}
        for joint, config in self.joint_profiles.items():
            neutral = config["neutral"]
            max_flex = config["max_flex"]
            
            # Linearly scale every finger and joint to the intent factor
            target_angles[joint] = neutral + (flex_factor * (max_flex - neutral))
            
        return target_angles

