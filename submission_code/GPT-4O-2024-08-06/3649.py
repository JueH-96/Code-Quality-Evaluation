class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        # Sort the strength array to break locks in increasing order of energy required
        strength.sort()
        
        # Initialize variables
        time = 0
        current_energy = 0
        current_factor = 1
        
        # Iterate over each lock
        for lock_strength in strength:
            # Calculate the time needed to reach the required energy for the current lock
            while current_energy < lock_strength:
                time += 1
                current_energy += current_factor
            
            # Break the lock, reset energy and increase factor
            current_energy = 0
            current_factor += K
        
        return time