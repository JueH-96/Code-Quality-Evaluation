from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        
        for i in range(n):
            # subarray must start with an even number within threshold
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                curr_len = 1
                prev_parity = nums[i] % 2
                # try to extend the subarray
                for j in range(i+1, n):
                    if nums[j] > threshold:
                        break
                    if nums[j] % 2 == prev_parity:
                        break
                    curr_len += 1
                    prev_parity = nums[j] % 2
                max_len = max(max_len, curr_len)
        
        return max_len