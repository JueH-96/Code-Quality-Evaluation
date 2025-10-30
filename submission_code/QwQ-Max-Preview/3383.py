from collections import defaultdict
from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        groups = defaultdict(list)
        for i in range(len(energy)):
            r = i % k
            groups[r].append(energy[i])
        
        max_energy = float('-inf')
        for group in groups.values():
            current_sum = 0
            current_max = float('-inf')
            for num in reversed(group):
                current_sum += num
                if current_sum > current_max:
                    current_max = current_sum
            if current_max > max_energy:
                max_energy = current_max
        return max_energy