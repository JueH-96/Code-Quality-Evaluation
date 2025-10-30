from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0

        for i in range(n):
            # The subarray must start with an even number and be <= threshold.
            if nums[i] % 2 != 0 or nums[i] > threshold:
                continue

            current_len = 1
            prev = nums[i]
            
            # Try to extend the subarray
            for j in range(i + 1, n):
                # Check if the current element meets the threshold requirement.
                if nums[j] > threshold:
                    break
                # Check if consecutive parity alternates.
                if prev % 2 == nums[j] % 2:
                    break
                # If valid, include the element in the subarray.
                current_len += 1
                prev = nums[j]
            
            # Update the maximum length found.
            max_len = max(max_len, current_len)
        
        return max_len