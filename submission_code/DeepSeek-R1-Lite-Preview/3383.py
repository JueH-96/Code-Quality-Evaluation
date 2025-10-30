from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        groups = [[] for _ in range(k)]
        
        # Group indices by i % k
        for i in range(n):
            r = i % k
            groups[r].append(i)
        
        max_sum = float('-inf')
        
        # Compute suffix sums for each group and track the maximum
        for r in range(k):
            if groups[r]:
                current_sum = 0
                for i in reversed(groups[r]):
                    current_sum += energy[i]
                    if current_sum > max_sum:
                        max_sum = current_sum
        
        return max_sum