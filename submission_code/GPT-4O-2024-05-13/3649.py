class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        total_time = 0
        current_factor = 1
        
        for s in strength:
            # Calculate the time needed to reach at least the required strength
            time_needed = (s + current_factor - 1) // current_factor
            total_time += time_needed
            # After breaking the lock, reset energy and increase the factor
            current_factor += K
        
        return total_time