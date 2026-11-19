"""
Dynamic CPU and process priority manager.
Can adjust thread count, process priority, and pauses on the fly while training.
"""

import os
import sys
import psutil
import torch
import time

class DynamicResourceManager:
    def __init__(self, cpu_threads: int = 4, priority: str = "normal", step_delay: float = 0.0):
        self.process = psutil.Process(os.getpid())
        self.cpu_threads = cpu_threads
        self.priority = priority
        self.step_delay = step_delay
        self.is_paused = False
        
        self.apply_cpu_threads(cpu_threads)
        self.apply_priority(priority)

    def apply_cpu_threads(self, threads: int):
        try:
            physical = psutil.cpu_count(logical=False) or threads
        except Exception:
            physical = threads
        # Выше физических ядер SMT только греет ноутбук, не ускоряя шаг.
        self.cpu_threads = max(1, min(threads, physical))
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, self.cpu_threads // 2)))
        except RuntimeError:
            pass # Interop threads can only be set once at startup

    def apply_priority(self, priority: str):
        self.priority = priority.lower()
        try:
            if sys.platform == "win32":
                prio_map = {
                    "idle": psutil.IDLE_PRIORITY_CLASS,
                    "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    "normal": psutil.NORMAL_PRIORITY_CLASS,
                    "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    "high": psutil.HIGH_PRIORITY_CLASS
                }
                if self.priority in prio_map:
                    self.process.nice(prio_map[self.priority])
            else:
                prio_map = {
                    "idle": 19,
                    "below_normal": 10,
                    "normal": 0,
                    "above_normal": -5,
                    "high": -10
                }
                if self.priority in prio_map:
                    self.process.nice(prio_map[self.priority])
        except Exception as e:
            # Permissions might restrict increasing priority
            pass

    def apply_preset(self, preset_dict: dict):
        if "cpu_threads" in preset_dict:
            self.apply_cpu_threads(preset_dict["cpu_threads"])
        if "process_priority" in preset_dict:
            self.apply_priority(preset_dict["process_priority"])
        if "step_delay_sec" in preset_dict:
            self.step_delay = preset_dict["step_delay_sec"]

    def throttle_step(self):
        """Called between training batches to prevent overheating and freezing other apps."""
        while self.is_paused:
            time.sleep(0.5)
            
        if self.step_delay > 0:
            time.sleep(self.step_delay)

    def get_stats(self) -> dict:
        try:
            mem = psutil.virtual_memory()
            proc_mem = self.process.memory_info().rss / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=None)
            return {
                "active_threads": self.cpu_threads,
                "priority": self.priority,
                "step_delay": self.step_delay,
                "ram_used_mb": proc_mem,
                "sys_ram_percent": mem.percent,
                "sys_cpu_percent": cpu_percent,
                "is_paused": self.is_paused
            }
        except Exception:
            return {}
