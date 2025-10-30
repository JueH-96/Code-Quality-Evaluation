class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        idx_1 = nums.index(1)
        idx_n = nums.index(n)

        if idx_1 < idx_n:
            return idx_1 + (n - 1 - idx_n)
        else:
            return idx_1 + (n - 1 - idx_n) - 1