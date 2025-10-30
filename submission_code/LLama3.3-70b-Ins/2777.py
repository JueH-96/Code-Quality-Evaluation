from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        
        for i in range(n):
            # Calculate the number of distinct elements in the prefix
            prefix_distinct = len(set(nums[:i+1]))
            
            # Calculate the number of distinct elements in the suffix
            suffix_distinct = len(set(nums[i+1:]))
            
            # Calculate the difference and append it to the result
            diff.append(prefix_distinct - suffix_distinct)
        
        return diff