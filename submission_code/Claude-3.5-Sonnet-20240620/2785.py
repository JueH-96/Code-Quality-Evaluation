class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        first = nums.index(1)
        last = nums.index(n)
        
        if first < last:
            return first + (n - 1 - last)
        else:
            return first + (n - 1 - last) - 1