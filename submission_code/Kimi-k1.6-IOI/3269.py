from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        # Generate the difference pattern based on consecutive elements in nums
        diff_pattern = []
        for i in range(len(nums) - 1):
            if nums[i+1] > nums[i]:
                diff_pattern.append(1)
            elif nums[i+1] == nums[i]:
                diff_pattern.append(0)
            else:
                diff_pattern.append(-1)
        
        # Calculate the number of valid subarrays by checking each possible window in diff_pattern
        m = len(pattern)
        count = 0
        n_diff = len(diff_pattern)
        for i in range(n_diff - m + 1):
            if diff_pattern[i:i+m] == pattern:
                count += 1
        return count