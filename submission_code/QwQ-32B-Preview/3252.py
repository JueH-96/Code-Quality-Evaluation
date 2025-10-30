from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute is_increasing_forward
        is_increasing_forward = [True] * n
        for i in range(1, n):
            if nums[i] <= nums[i-1]:
                is_increasing_forward[i] = False
            else:
                is_increasing_forward[i] = is_increasing_forward[i-1]
        
        # Precompute is_increasing_backward
        is_increasing_backward = [True] * n
        for i in range(n-2, -1, -1):
            if nums[i] >= nums[i+1]:
                is_increasing_backward[i] = False
            else:
                is_increasing_backward[i] = is_increasing_backward[i+1]
        
        count = 0
        for start in range(n):
            for end in range(start, n):
                if start == 0:
                    if end+1 >= n or is_increasing_backward[end+1]:
                        count += 1
                elif end == n-1:
                    if start-1 < 0 or is_increasing_forward[start-1]:
                        count += 1
                else:
                    if (start-1 < 0 or is_increasing_forward[start-1]) and \
                       (end+1 >= n or is_increasing_backward[end+1]) and \
                       (start-1 < 0 or end+1 >= n or nums[start-1] < nums[end+1]):
                        count += 1
        return count