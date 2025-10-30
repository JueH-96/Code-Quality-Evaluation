class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        # Find the positions of 1 and n in the permutation
        pos_1 = nums.index(1)
        pos_n = nums.index(len(nums))
        
        # The minimum number of operations needed is the sum of the distances
        # of 1 and n from their correct positions at the start and end of the list
        return pos_1 + (len(nums) - 1 - pos_n)