from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Check if all elements are the same
        if all(x == nums[0] for x in nums):
            return 0
        # Create a dictionary to map each value to its indices
        value_indices = defaultdict(list)
        for i, num in enumerate(nums):
            value_indices[num].append(i)
        min_steps = float('inf')
        for val, indices in value_indices.items():
            m = len(indices)
            if m == 1:
                # Only one occurrence, step is n//2
                current_max = n // 2
            else:
                current_max = 0
                for i in range(m):
                    current = indices[i]
                    next_idx = indices[(i + 1) % m]
                    distance = (next_idx - current) % n
                    step = distance // 2
                    if step > current_max:
                        current_max = step
            if current_max < min_steps:
                min_steps = current_max
        return min_steps