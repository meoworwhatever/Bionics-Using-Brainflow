import time

class SafetyGuard:
    def __init__(self):
        
        self.MIN_ANGLE = 0
        self.MAX_ANGLE = 160  
        self.MAX_VELOCITY = 45 
        
        self.ARTIFACT_THRESHOLD = 150.0 
        self.last_position = 0
        self.last_timestamp = time.time()

    def validate_movement(self, target_position, current_rms):
        """
        The ultimate 'Gatekeeper' function. 
        Returns (is_safe: bool, safe_position: float)
        """
        now = time.time()
        dt = now - self.last_timestamp
        
        if current_rms > self.ARTIFACT_THRESHOLD:
            print("SAFETY: Neural artifact detected. Freezing limb.")
            return False, self.last_position

        clamped_position = max(self.MIN_ANGLE, min(target_position, self.MAX_ANGLE))

        if dt > 0:
            velocity = abs(clamped_position - self.last_position) / dt
            if velocity > self.MAX_VELOCITY:
                direction = 1 if (clamped_position > self.last_position) else -1
                clamped_position = self.last_position + (direction * self.MAX_VELOCITY * dt)
                print(f"SAFETY: Velocity limit hit. Slowing move.")

        self.last_position = clamped_position
        self.last_timestamp = now
        
        return True, round(clamped_position, 2)

    def emergency_stop(self):
        """Call this if the SocketIO connection drops."""
        print("EMERGENCY STOP ACTIVATED")
        return 0
