from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the overall dominant element using Boyer-Moore majority vote
        candidate = None
        count = 0
        for x in nums:
            if count == 0:
                candidate = x
                count = 1
            else:
                count += 1 if x == candidate else -1
        
        # Count total occurrences of the candidate
        total = nums.count(candidate)
        
        # Now scan for the smallest split index
        prefix_count = 0
        for i in range(n - 1):
            if nums[i] == candidate:
                prefix_count += 1
            prefix_len = i + 1
            suffix_len = n - prefix_len
            suffix_count = total - prefix_count
            
            # Check both subarrays have the same dominant element
            if prefix_count * 2 > prefix_len and suffix_count * 2 > suffix_len:
                return i
        
        return -1