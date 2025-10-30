from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        pos = {}
        
        # Store positions of each number
        for i, num in enumerate(nums):
            if num not in pos:
                pos[num] = []
            pos[num].append(i)
        
        min_time = float('inf')
        
        # Calculate minimum time for each unique number
        for indices in pos.values():
            max_gap = max(indices[0] + n - indices[-1], max(indices[i+1] - indices[i] for i in range(len(indices) - 1))) // 2
            min_time = min(min_time, max_gap)
        
        return min_time