class Solution:
    def minimumOperations(self, nums, target):
        return sum(abs(a - b) for a, b in zip(nums, target))