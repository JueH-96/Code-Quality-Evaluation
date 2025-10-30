class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        i_1 = nums.index(1)
        i_n = nums.index(n)
        return i_1 + (n - 1 - i_n) - (1 if i_1 > i_n else 0)