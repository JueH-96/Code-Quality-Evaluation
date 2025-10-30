class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find positions of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # Calculate swaps needed
        # Swaps to move 1 to front
        swaps_for_1 = pos1
        
        # Swaps to move n to end
        swaps_for_n = (n - 1) - posn
        
        # If 1 comes after n, they will cross paths and we save one swap
        if pos1 > posn:
            return swaps_for_1 + swaps_for_n - 1
        else:
            return swaps_for_1 + swaps_for_n