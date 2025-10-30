from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        one_index, n_index = nums.index(1), nums.index(n)
        operations = 0
        
        if one_index > n_index:
            operations += one_index + (n - 1 - n_index)
        else:
            operations += one_index + (n - 1 - n_index) - (n_index - one_index - 1)
        
        return operations