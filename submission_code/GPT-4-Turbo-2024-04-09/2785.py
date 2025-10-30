class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the index of 1 and n
        index_1 = nums.index(1)
        index_n = nums.index(n)
        
        # Calculate the number of swaps needed to bring 1 to the start
        swaps_for_1 = index_1  # As we need to swap 1 leftwards index_1 times
        
        # Calculate the number of swaps needed to bring n to the end
        swaps_for_n = (n - 1) - index_n  # As we need to swap n rightwards (n - 1 - index_n) times
        
        return swaps_for_1 + swaps_for_n