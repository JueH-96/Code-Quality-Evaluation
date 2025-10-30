class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posN = nums.index(n)
        total_swaps = pos1 + (n - 1 - posN)
        if pos1 > posN:
            total_swaps -= 1
        return total_swaps