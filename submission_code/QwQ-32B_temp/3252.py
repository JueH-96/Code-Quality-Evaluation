from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # As per constraints, this case won't occur
        
        # Precompute left_incr array
        left_incr = [True] * (n + 1)
        for i in range(1, n + 1):
            if i == 1:
                left_incr[i] = True
            else:
                left_incr[i] = left_incr[i-1] and (nums[i-2] < nums[i-1])
        
        # Precompute right_incr array
        right_incr = [False] * n
        right_incr[-1] = True  # single element is valid
        for s in range(n-2, -1, -1):
            if nums[s] < nums[s+1] and right_incr[s+1]:
                right_incr[s] = True
            else:
                right_incr[s] = False
        
        count = 0
        for start in range(n):
            for end in range(start, n):
                # Determine left and right validity
                left_valid = left_incr[start]
                right_start = end + 1
                if right_start < n:
                    right_valid = right_incr[right_start]
                else:
                    right_valid = True
                
                if not (left_valid and right_valid):
                    continue
                
                # Check conditions for the combined array
                if start == 0 and end == n - 1:
                    count += 1
                    continue
                if start == 0 or end == n - 1:
                    count += 1
                else:
                    if nums[start - 1] < nums[end + 1]:
                        count += 1
        
        return count