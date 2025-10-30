class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0]*n
        
        for i in range(n):
            prefix_distinct = len(set(nums[:i+1]))
            suffix_distinct = len(set(nums[i+1:]))
            diff[i] = prefix_distinct - suffix_distinct
        
        return diff