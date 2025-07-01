import math
import time
import random

# --- Utility Class ---
class point_box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def calc_bbox(self):
        return (self.x - self.w/2, self.y - self.h/2, self.x + self.w/2, self.y + self.h/2)
    
    def random_point(self):
        mx, my, Mx, My = self.calc_bbox()
        return [int(random.uniform(mx, Mx)), int(random.uniform(my, My))]

# --- Animation Calculation Classes ---

class BaseAnim:
    current_time = time.time()
    
    @classmethod
    def update_time(cls):
        cls.current_time = time.time()

    def __init__(self):
        self.start_time = 0 
        self.is_active = False 

    def activate(self, current_time):
        if not self.is_active:
            self.start_time = current_time
            self.is_active = True

    def deactivate(self):
        self.is_active = False
        self.start_time = 0

class MovementAnim(BaseAnim):
    def __init__(self, velocity=1, angle=0, hz=3, amplitude=10):
        super().__init__()
        self.vel = velocity
        raw_angle = random.randrange(-abs(angle) - 1, abs(angle) + 1) + round(random.random(), 2)
        self.angle_rad = math.radians(raw_angle)
        self.hz = hz
        self.amplitude = amplitude
        self.rand_vibe = round(random.random() * 10, 2)

    def _get_elapsed_time(self):
        if not self.is_active:
            return 0
        return self.current_time - self.start_time

    def move_right(self):
        distance = self.vel * self._get_elapsed_time()
        return (distance * math.cos(self.angle_rad), distance * math.sin(self.angle_rad))

    def move_left(self):
        distance = self.vel * self._get_elapsed_time()
        return (-distance * math.cos(self.angle_rad), -distance * math.sin(self.angle_rad))

    def move_up(self):
        distance = self.vel * self._get_elapsed_time()
        return (distance * math.sin(self.angle_rad), -distance * math.cos(self.angle_rad))

    def move_down(self):
        distance = self.vel * self._get_elapsed_time()
        return (-distance * math.sin(self.angle_rad), distance * math.cos(self.angle_rad))
    
    def vibration_side(self):
        if not self.is_active:
            return (0, 0)
        # Using current_time for continuous vibration, not elapsed time for accumulating movement
        offset = self.amplitude * math.sin(2 * math.pi * self.hz * self.current_time + self.rand_vibe) 
        return (offset * math.cos(self.angle_rad), offset * math.sin(self.angle_rad))

    def vibration_updown(self):
        if not self.is_active:
            return (0, 0)
        # Using current_time for continuous vibration, not elapsed time for accumulating movement
        offset = self.amplitude * math.sin(2 * math.pi * self.hz * self.current_time + self.rand_vibe) 
        return (offset * -math.sin(self.angle_rad), offset * math.cos(self.angle_rad))

class OpacityAnim(BaseAnim):
    # Changed parameters: Removed fade_ratio, added is_fade_in
    def __init__(self, life_time=1.0, alpha=1.0, smooth=False, is_fade_in=True):
        super().__init__()
        if not (0 <= alpha <= 1):
            raise ValueError(f"Max alpha value must be between 0 and 1. Current value: {alpha}")
        
        self.life_time = life_time
        self.max_alpha = alpha
        self.is_fade_in = is_fade_in # True for fade-in, False for fade-out
        
        # alpha_func now represents a transition from 1.0 to 0.0 over its input range
        self.alpha_func = self._smooth_alpha if smooth else self._linear_alpha

    def _smooth_alpha(self, current, total):
        if total <= 0: return 0.0
        # Returns a value from 1.0 down to 0.0 as current goes from 0 to total
        return 1.0 - (current / total) ** 2 

    def _linear_alpha(self, current, total):
        if total <= 0: return 0.0
        # Returns a value from 1.0 down to 0.0 as current goes from 0 to total
        return max(0.0, 1.0 - current / total)

    def get_alpha(self):
        if not self.is_active:
            return 0.0

        elapsed_time = self.current_time - self.start_time
        
        if elapsed_time < 0: return 0.0 
        if elapsed_time >= self.life_time: return 0.0 # Animation is over

        progress = elapsed_time / self.life_time # Progress from 0.0 to 1.0 over life_time

        if self.is_fade_in:
            # For fade-in: alpha should increase from 0 to max_alpha
            if self.alpha_func.__name__ == '_linear_alpha':
                # Linear fade-in: directly use progress
                return self.max_alpha * progress
            else: # Smooth fade-in: reverse the smooth_alpha function
                # _smooth_alpha goes from 1.0 to 0.0. (1.0 - _smooth_alpha) will go from 0.0 to 1.0
                return self.max_alpha * (1.0 - self.alpha_func(elapsed_time, self.life_time))
        else: # is_fade_out
            # For fade-out: alpha should decrease from max_alpha to 0
            # Our alpha_func already goes from 1.0 to 0.0. So, we just apply it.
            return self.max_alpha * self.alpha_func(elapsed_time, self.life_time)

# --- Animation Wrapper Classes ---
class Movement:
    def __init__(self, anim, duration=1.0, vel=1, angle=0):
        self.type = "movement"
        self.anim = anim # "up", "down", "left", "right"
        self.params = {"duration": duration, "vel": vel, "angle": angle}
        self.duration = duration
        self.vel = vel
        self.angle = angle

    def clone(self):
        return Movement(self.anim, self.duration, self.vel, self.angle)
    
class Vibration:
    def __init__(self, anim, duration=1.0, hz=3, amp=10):
        self.type = "vibration"
        self.anim = anim # "side", "updown"
        self.params = {"duration": duration, "hz": hz, "amp": amp}
        self.duration = duration
        self.hz = hz
        self.amp = amp

    def clone(self):
        return Vibration(self.anim, self.duration, self.hz, self.amp)
    
class Fade:
    # Changed parameters: Added anim, removed fade_ratio
    def __init__(self, anim="in", duration=1.0, alpha=1.0, smooth=False):
        self.type = "fade"
        # Validate anim type
        if anim not in ["in", "out"]:
            raise ValueError("Fade 'anim' must be 'in' or 'out'.")
        
        self.anim = anim # "in" or "out"
        self.params = {"lt": duration, "A": alpha, "smooth": smooth}
        self.duration = duration
        self.alpha = alpha
        self.smooth = smooth

    # 복사 생성자
    def clone(self):
        return Fade(self.anim, self.duration, self.alpha, self.smooth)

# --- Animation Executor Classes ---

class Sequential:
    def __init__(self, steps):
        self.steps = steps
        self.current_step_index = 0
        self.step_start_time = 0
        self._active_anim = None
        self.cumulative_pos_offset = (0, 0)
        self._setup_step()

    def _setup_step(self):
        if self._active_anim and hasattr(self._active_anim, 'deactivate'):
            self._active_anim.deactivate()

        if self.is_finished():
            self._active_anim = None
            return

        step = self.steps[self.current_step_index]
        self.duration = 0 
        self.step_start_time = BaseAnim.current_time 

        if isinstance(step, Parallel): 
            self._active_anim = step
            self._active_method = self._active_anim.step
            self.duration = step.max_duration
            self._active_anim.activate(self.step_start_time)
            return

        p = step.params
        
        if step.type == "movement":
            self._active_anim = MovementAnim(velocity=p["vel"], angle=p["angle"])
            self._active_method = getattr(self._active_anim, f'move_{step.anim}')
            self.duration = p["duration"]
        elif step.type == "vibration":
            self._active_anim = MovementAnim(hz=p["hz"], amplitude=p["amp"])
            self._active_method = getattr(self._active_anim, f'vibration_{step.anim}')
            self.duration = p["duration"]
        elif step.type == "fade":
            # Pass is_fade_in based on step.anim
            is_fade_in_flag = (step.anim == "in")
            self._active_anim = OpacityAnim(life_time=p["lt"], alpha=p["A"], smooth=p.get("smooth", False), is_fade_in=is_fade_in_flag)
            self._active_method = self._active_anim.get_alpha
            self.duration = p["lt"]
        
        if self._active_anim:
            self._active_anim.activate(self.step_start_time)

    def is_finished(self):
        return self.current_step_index >= len(self.steps)

    def step(self):
        BaseAnim.update_time()
        
        if self.is_finished():
            return {"pos_offset": self.cumulative_pos_offset, "alpha": 0.0}

        if self.duration > 0 and (BaseAnim.current_time - self.step_start_time) >= self.duration:
            original_time = BaseAnim.current_time
            BaseAnim.current_time = self.step_start_time + self.duration
            
            final_step_result = self._active_method()
            
            if self._active_anim and hasattr(self._active_anim, 'deactivate'):
                self._active_anim.deactivate()

            BaseAnim.current_time = original_time 

            step_pos_offset = (0, 0)
            if isinstance(final_step_result, dict):
                step_pos_offset = final_step_result.get('pos_offset', (0, 0))
            elif isinstance(final_step_result, tuple):
                step_pos_offset = final_step_result
            
            self.cumulative_pos_offset = (self.cumulative_pos_offset[0] + step_pos_offset[0],
                                          self.cumulative_pos_offset[1] + step_pos_offset[1])

            self.current_step_index += 1
            self._setup_step()
            
            return self.step()
        
        result = self._active_method()
        
        current_step_offset = (0, 0)
        current_alpha = 1.0

        if isinstance(self._active_anim, Parallel):
            current_step_offset = result["pos_offset"]
            current_alpha = result["alpha"]
        elif isinstance(self._active_anim, MovementAnim):
            current_step_offset = result
        elif isinstance(self._active_anim, OpacityAnim):
            current_alpha = result

        total_offset = (self.cumulative_pos_offset[0] + current_step_offset[0],
                        self.cumulative_pos_offset[1] + current_step_offset[1])
        
        return {"pos_offset": total_offset, "alpha": current_alpha}
    
    def clone(self):
        return Sequential([step.clone() for step in self.steps])

class Parallel:
    def __init__(self, steps):
        self.steps = steps
        self.exec_anims = []
        self.start_time = 0 
        self.max_duration = 0 
        self.is_active = False 

        for step in steps:
            p = step.params
            anim_obj, method = None, None
            duration = 0 

            if step.type == "movement":
                anim_obj = MovementAnim(velocity=p["vel"], angle=p["angle"])
                method = getattr(anim_obj, f'move_{step.anim}')
                duration = p["duration"]
            elif step.type == "vibration":
                anim_obj = MovementAnim(hz=p["hz"], amplitude=p["amp"])
                method = getattr(anim_obj, f'vibration_{step.anim}')
                duration = p["duration"]
            elif step.type == "fade":
                # Pass is_fade_in based on step.anim
                is_fade_in_flag = (step.anim == "in")
                anim_obj = OpacityAnim(life_time=p["lt"], alpha=p["A"], smooth=p.get("smooth", False), is_fade_in=is_fade_in_flag)
                method = anim_obj.get_alpha
                duration = p["lt"] 
            
            if anim_obj and method:
                self.exec_anims.append({"obj": anim_obj, "method": method, "duration": duration})
                self.max_duration = max(self.max_duration, duration) 
    
    def activate(self, current_time):
        if not self.is_active:
            self.start_time = current_time
            self.is_active = True
            for anim_info in self.exec_anims:
                anim_info["obj"].activate(current_time) 

    def deactivate(self):
        self.is_active = False
        self.start_time = 0
        for anim_info in self.exec_anims:
            anim_info["obj"].deactivate()

    def is_finished(self):
        if not self.is_active:
            return True
        return (BaseAnim.current_time - self.start_time) >= self.max_duration

    def step(self):
        BaseAnim.update_time()
        
        total_dx, total_dy = 0, 0
        final_alpha = 1.0 
        has_fade = any(isinstance(anim_info["obj"], OpacityAnim) for anim_info in self.exec_anims)

        if self.is_finished():
            original_current_time = BaseAnim.current_time
            BaseAnim.current_time = self.start_time + self.max_duration 
            
            current_dx, current_dy = 0, 0
            current_alpha = 1.0
            
            for anim_info in self.exec_anims:
                obj = anim_info["obj"]
                method = anim_info["method"]
                
                was_active = obj.is_active
                if not was_active:
                    obj.activate(self.start_time) 
                
                obj.start_time = self.start_time 

                result = method() 

                if not was_active:
                    obj.deactivate()

                if isinstance(obj, MovementAnim):
                    dx, dy = result
                    current_dx += dx
                    current_dy += dy
                elif isinstance(obj, OpacityAnim):
                    current_alpha = min(current_alpha, result) 

            BaseAnim.current_time = original_current_time 
            
            return {"pos_offset": (current_dx, current_dy), "alpha": current_alpha if has_fade else 0.0}

        for anim_info in self.exec_anims:
            obj = anim_info["obj"]
            method = anim_info["method"]
            result = method() 

            if isinstance(obj, MovementAnim):
                dx, dy = result
                total_dx += dx
                total_dy += dy
            elif isinstance(obj, OpacityAnim):
                final_alpha = min(final_alpha, result)
        
        if not has_fade:
            final_alpha = 1.0

        return {"pos_offset": (total_dx, total_dy), "alpha": final_alpha}
    
    def clone(self):
        return Parallel([step.clone() for step in self.steps])