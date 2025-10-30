from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        best = 0
        
        for l in range(n):
            # The subarray must start with an even number and be ≤ threshold
            if nums[l] % 2 or nums[l] > threshold:
                continue
            
            # At least the starting element itself is a valid subarray
            best = max(best, 1)
            curr_len = 1
            
            # Try to extend the subarray to the right
            for r in range(l + 1, n):
                # Condition 3: element must be ≤ threshold
                if nums[r] > threshold:
                    break
                
                # Condition 2: parity must alternate with previous element
                if (nums[r] % 2) == (nums[r - 1] % 2):
                    break
                
                curr_len += 1
                best = max(best, curr_len)
        
        return best