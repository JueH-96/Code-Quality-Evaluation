class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_of_1 = nums.index(1)
        index_of_n = nums.index(n)
        
        if index_of_1 < index_of_n:
            return index_of_1 + (n - 1 - index_of_n)
        else:
            return index_of_1 + (n - 1 - index_of_n) - 1