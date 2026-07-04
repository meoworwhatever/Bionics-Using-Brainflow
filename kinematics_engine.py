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
