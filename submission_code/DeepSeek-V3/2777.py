class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            prefix_unique = len(set(prefix))
            suffix_unique = len(set(suffix))
            diff.append(prefix_unique - suffix_unique)
        return diff