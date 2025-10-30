from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        
        for i in range(n):
            # Start only on even nums[i] and within threshold
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                curr_len = 1
                prev_parity = nums[i] % 2
                
                # Try to extend the subarray from i+1 onwards
                for j in range(i + 1, n):
                    if nums[j] <= threshold and nums[j] % 2 != prev_parity:
                        curr_len += 1
                        prev_parity = nums[j] % 2
                    else:
                        break
                
                max_len = max(max_len, curr_len)
        
        return max_len