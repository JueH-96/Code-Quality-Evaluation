class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index1 = nums.index(1)
        indexn = nums.index(n)
        if index1 < indexn:
            return index1 + (n - 1 - indexn)
        else:
            return index1 + (n - 1 - indexn) - 1