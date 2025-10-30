from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        for start in range(k):
            current_energy = 0
            for i in range(start, n, k):
                current_energy += energy[i]
            max_energy = max(max_energy, current_energy)
        
        return max_energy