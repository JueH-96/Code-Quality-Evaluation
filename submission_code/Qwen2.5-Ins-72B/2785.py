class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_1 = nums.index(1)
        index_n = nums.index(n)
        
        # If 1 is already at the start or n is already at the end, no need to swap them
        if index_1 == 0 and index_n == n - 1:
            return 0
        
        # Calculate the number of swaps needed to move 1 to the start
        swaps_to_move_1 = index_1
        
        # Calculate the number of swaps needed to move n to the end
        swaps_to_move_n = n - 1 - index_n
        
        # If 1 is to the right of n, we need to subtract one swap because the last swap of 1 will also move n one step closer
        if index_1 > index_n:
            return swaps_to_move_1 + swaps_to_move_n - 1
        else:
            return swaps_to_move_1 + swaps_to_move_n