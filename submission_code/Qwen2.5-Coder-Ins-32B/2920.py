from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        positions = defaultdict(list)
        
        # Record the positions of each number
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        min_seconds = float('inf')
        
        # Calculate the minimum seconds needed for each unique number
        for indices in positions.values():
            max_distance = 0
            for i in range(len(indices)):
                # Calculate the distance to the next occurrence, considering wrap-around
                distance = (indices[(i + 1) % len(indices)] - indices[i] + n) % n
                max_distance = max(max_distance, distance)
            # The time needed to spread the number is half the maximum distance
            min_seconds = min(min_seconds, max_distance // 2)
        
        return min_seconds