import bisect
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        next_greater = [n] * n
        stack = []
        
        # Compute next greater element for each index
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                next_greater[i] = stack[-1]
            else:
                next_greater[i] = n
            stack.append(i)
        
        # Build positions dictionary
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        count = 0
        for i in range(n):
            current_num = nums[i]
            ng = next_greater[i]
            L = i
            R = ng - 1
            
            indices = pos[current_num]
            left = bisect.bisect_left(indices, L)
            right_idx = bisect.bisect_right(indices, R)
            count += (right_idx - left)
        
        return count