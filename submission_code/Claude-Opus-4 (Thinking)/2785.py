class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # Calculate total swaps needed
        result = pos1 + (n - 1 - posn)
        
        # If 1 is to the right of n, they will cross paths
        if pos1 > posn:
            result -= 1
            
        return result