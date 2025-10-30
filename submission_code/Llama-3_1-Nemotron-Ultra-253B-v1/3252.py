from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute left_valid: left_valid[i] indicates if the subarray [0..i-1] is strictly increasing
        left_valid = [False] * (n + 1)
        left_valid[0] = True
        for i in range(1, n + 1):
            if i == 1:
                left_valid[i] = True
            else:
                left_valid[i] = left_valid[i-1] and (nums[i-2] < nums[i-1])
        
        # Precompute right_valid: right_valid[j] indicates if the subarray [j+1..n-1] is strictly increasing
        right_valid = [False] * n
        right_valid[-1] = True
        for j in range(n-2, -1, -1):
            if j + 1 == n - 1:
                right_valid[j] = True
            else:
                right_valid[j] = right_valid[j+1] and (nums[j+1] < nums[j+2])
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                if left_valid[i] and right_valid[j]:
                    # Check the merge condition
                    if i == 0 or j == n-1 or (nums[i-1] < nums[j+1]):
                        count += 1
        
        return count