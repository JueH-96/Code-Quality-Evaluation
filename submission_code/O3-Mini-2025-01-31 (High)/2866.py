from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        for i in range(n):
            # The subarray must start with an even number and be <= threshold.
            if nums[i] % 2 != 0 or nums[i] > threshold:
                continue
            
            current_length = 1
            prev_parity = nums[i] % 2
            
            # Try to extend the subarray as long as conditions hold.
            for j in range(i + 1, n):
                if nums[j] > threshold:
                    break
                # Condition: alternating parity 
                if nums[j] % 2 == prev_parity:
                    break
                current_length += 1
                prev_parity = nums[j] % 2
            
            max_length = max(max_length, current_length)
        return max_length