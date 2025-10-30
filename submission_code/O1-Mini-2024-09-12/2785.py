class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        idx1 = nums.index(1)
        idxn = nums.index(n)
        if idxn < idx1:
            return idx1 + (n - idxn - 2)
        else:
            return idx1 + (n - idxn - 1)