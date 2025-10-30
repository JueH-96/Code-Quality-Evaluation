class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        if pos1 < posn:
            return pos1 + (n - 1 - posn)
        else:
            return pos1 + (n - 1 - posn) - 1