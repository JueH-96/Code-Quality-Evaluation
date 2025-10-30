from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')

        for i in range(n):
            current_energy = 0
            j = i
            while j < n:
                current_energy += energy[j]
                j += k
            max_energy = max(max_energy, current_energy)

        return max_energy