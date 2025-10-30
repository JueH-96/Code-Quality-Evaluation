from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        m = len(pattern)
        
        # Iterate over the array with a sliding window of size m + 1
        for i in range(len(nums) - m):
            match = True
            
            # Check if the current subarray matches the pattern
            for j in range(m):
                if pattern[j] == 1 and nums[i + j + 1] <= nums[i + j]:
                    match = False
                    break
                elif pattern[j] == 0 and nums[i + j + 1] != nums[i + j]:
                    match = False
                    break
                elif pattern[j] == -1 and nums[i + j + 1] >= nums[i + j]:
                    match = False
                    break
            
            # If the subarray matches the pattern, increment the count
            if match:
                count += 1
        
        return count