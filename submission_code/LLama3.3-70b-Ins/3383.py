from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        # Try all possible starting points
        for i in range(n):
            current_energy = 0
            j = i
            # Keep jumping and absorbing energy until we reach the end
            while j < n:
                current_energy += energy[j]
                j += k
            # Update the maximum energy
            max_energy = max(max_energy, current_energy)
        
        return max_energy