class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        if pos1 < posn:
            new_posn = posn - pos1
        else:
            new_posn = posn + (pos1 - posn)
        
        return pos1 + (n - 1 - new_posn)