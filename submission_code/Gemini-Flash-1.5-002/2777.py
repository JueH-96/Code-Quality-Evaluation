class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        for i in range(n):
            prefix = set(nums[:i+1])
            suffix = set(nums[i+1:])
            diff.append(len(prefix) - len(suffix))
        return diff