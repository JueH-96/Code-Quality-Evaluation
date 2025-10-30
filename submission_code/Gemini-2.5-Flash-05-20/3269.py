from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        
        count = 0
        
        # Iterate through all possible starting indices 'i' for a subarray in nums
        # that could potentially match the pattern.
        # A matching subarray has length m + 1. If it starts at 'i', it ends at 'i + m'.
        # The last index 'i + m' must be less than 'n'.
        # So, 'i' can range from 0 up to n - m - 1 (inclusive).
        # This is equivalent to range(n - m).
        for i in range(n - m):
            current_subarray_matches = True
            
            # For each potential starting index 'i', check if the relationships
            # between adjacent elements in nums[i ... i + m] match the pattern.
            # The pattern has 'm' elements, pattern[0] through pattern[m-1].
            # pattern[k] describes the relationship between nums[i + k] and nums[i + k + 1].
            for k in range(m):
                val1 = nums[i + k]
                val2 = nums[i + k + 1]
                
                if pattern[k] == 1:
                    # Expected: nums[i + k + 1] > nums[i + k]
                    if not (val2 > val1):
                        current_subarray_matches = False
                        break  # Mismatch found, no need to check further for this 'i'
                elif pattern[k] == 0:
                    # Expected: nums[i + k + 1] == nums[i + k]
                    if not (val2 == val1):
                        current_subarray_matches = False
                        break  # Mismatch found
                elif pattern[k] == -1:
                    # Expected: nums[i + k + 1] < nums[i + k]
                    if not (val2 < val1):
                        current_subarray_matches = False
                        break  # Mismatch found
            
            # If all conditions for the current 'i' were met (no mismatch found in the inner loop),
            # increment the total count of matching subarrays.
            if current_subarray_matches:
                count += 1
                
        return count