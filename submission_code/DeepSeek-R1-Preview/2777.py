class Solution:
    def distinctDifferenceArray(self, nums):
        diff = []
        n = len(nums)
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            prefix_count = len(set(prefix))
            suffix_count = len(set(suffix))
            diff.append(prefix_count - suffix_count)
        return diff