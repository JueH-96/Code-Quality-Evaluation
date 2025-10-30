class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the index of 1 and n
        index_1 = nums.index(1)
        index_n = nums.index(n)
        
        # If 1 is before n, we can directly swap them to the start and end
        if index_1 < index_n:
            return index_1 + (n - 1 - index_n)
        else:
            # If 1 is after n, we need one extra swap because they will cross each other
            return index_1 + (n - 1 - index_n) - 1