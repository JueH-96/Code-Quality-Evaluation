from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        
        for l in range(n):
            # Start only if first element is even and within threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                curr_len = 1
                prev_parity = nums[l] % 2
                # Try to extend the subarray from l+1 onward
                for r in range(l+1, n):
                    if nums[r] > threshold:
                        break
                    curr_parity = nums[r] % 2
                    # Check alternating parity
                    if curr_parity == prev_parity:
                        break
                    curr_len += 1
                    prev_parity = curr_parity
                max_len = max(max_len, curr_len)
        
        return max_len