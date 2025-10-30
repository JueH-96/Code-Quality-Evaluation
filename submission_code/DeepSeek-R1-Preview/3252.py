from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute is_increasing_prefix: whether nums[0..i-1] is strictly increasing
        is_increasing_prefix = [True] * (n + 1)
        for i in range(2, n + 1):
            is_increasing_prefix[i] = is_increasing_prefix[i-1] and (nums[i-2] < nums[i-1])
        
        # Precompute right_valid: whether nums[j+1..n-1] is strictly increasing
        right_valid = [True] * n
        for j in range(n-2, -1, -1):
            if j+1 == n-1:
                right_valid[j] = True
            else:
                right_valid[j] = (nums[j+1] < nums[j+2]) and right_valid[j+1]
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                if is_increasing_prefix[i] and right_valid[j]:
                    if i == 0 and j == n-1:
                        count += 1
                    else:
                        if i == 0:
                            count += 1
                        elif j == n-1:
                            count += 1
                        else:
                            if nums[i-1] < nums[j+1]:
                                count += 1
        return count