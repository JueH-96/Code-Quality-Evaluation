from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        strength.sort()  # Sort the strengths to break the weakest locks first
        
        time = 0
        current_energy = 0
        current_factor = 1
        
        for s in strength:
            # Increase time until current_energy is enough to break the lock
            while current_energy < s:
                time += 1
                current_energy += current_factor
            
            # After breaking the lock, reset energy and increase factor
            current_energy = 0
            current_factor += K
        
        return time