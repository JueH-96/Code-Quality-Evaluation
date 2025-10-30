import bisect
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute left boundaries (previous greater element)
        left_boundaries = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left_boundaries[i] = stack[-1]
            else:
                left_boundaries[i] = -1
            stack.append(i)
        
        # Precompute right boundaries (next greater element)
        right_boundaries = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                right_boundaries[i] = stack[-1]
            else:
                right_boundaries[i] = n
            stack.append(i)
        
        # Group positions by their value
        pos_map = defaultdict(list)
        for idx, num in enumerate(nums):
            pos_map[num].append(idx)
        
        total = 0
        
        for x in pos_map:
            pos = pos_map[x]
            m = len(pos)
            for k in range(m):
                pk = pos[k]
                R = right_boundaries[pk]
                max_j_val = R - 1
                # Find the largest index t where pos[t] <= max_j_val
                t = bisect.bisect_right(pos, max_j_val) - 1
                if t >= k:
                    total += t - k + 1
        
        return total