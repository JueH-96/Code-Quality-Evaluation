class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        one_index = nums.index(1)
        n_index = nums.index(n)
        
        if one_index < n_index:
            return one_index + (n - 1 - n_index)
        else:
            return one_index + (n - 1 - n_index) - 1