from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        max_total = -float('inf')
        for r in range(k):
            chain = energy[r::k]
            current_sum = 0
            max_suffix = -float('inf')
            for num in reversed(chain):
                current_sum += num
                if current_sum > max_suffix:
                    max_suffix = current_sum
            if max_suffix > max_total:
                max_total = max_suffix
        return max_total