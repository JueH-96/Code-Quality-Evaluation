from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        # Find the index of 1 and n in the array
        index_1 = nums.index(1)
        index_n = nums.index(len(nums))
        
        # If 1 is already at the beginning and n is already at the end, return 0
        if index_1 == 0 and index_n == len(nums) - 1:
            return 0
        
        # If 1 is at the end and n is at the beginning, we need to swap them first
        if index_1 == len(nums) - 1 and index_n == 0:
            return len(nums) - 2
        
        # If 1 is not at the beginning, we need to move it to the beginning
        # If n is not at the end, we need to move it to the end
        # The minimum number of operations is the sum of the distances from 1 to the beginning and from n to the end
        return index_1 + (len(nums) - 1 - index_n)