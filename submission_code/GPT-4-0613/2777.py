class Solution:
    def distinctDifferenceArray(self, nums):
        n = len(nums)
        prefix = [0] * n
        suffix = [0] * n
        diff = [0] * n
        prefix_set = set()
        suffix_set = set(nums)
        
        for i in range(n):
            prefix_set.add(nums[i])
            suffix_set.remove(nums[i])
            prefix[i] = len(prefix_set)
            suffix[i] = len(suffix_set)
        
        for i in range(n):
            diff[i] = prefix[i] - suffix[i]
        
        return diff