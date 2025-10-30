class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos_1 = nums.index(1)
        pos_n = nums.index(n)
        swaps = pos_1 + (n - 1 - pos_n)
        if pos_1 > pos_n:
            swaps -= 1
        return swaps