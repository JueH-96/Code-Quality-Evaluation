class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        pos1 = nums.index(1)
        posn = nums.index(len(nums))
        if pos1 < posn:
            return pos1 + (len(nums)-1 - posn)
        else:
            return pos1 + (len(nums)-1 - posn) - 1