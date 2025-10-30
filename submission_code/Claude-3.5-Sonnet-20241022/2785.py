class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find positions of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # If already in correct positions, return 0
        if pos1 == 0 and posn == n-1:
            return 0
            
        # Calculate minimum swaps needed
        # If 1 is after n, we need one less swap since they will cross
        if pos1 > posn:
            return pos1 + (n-1-posn) - 1
        else:
            return pos1 + (n-1-posn)