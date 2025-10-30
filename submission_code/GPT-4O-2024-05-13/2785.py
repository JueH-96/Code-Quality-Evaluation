class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # If 1 is already at the start and n is already at the end
        if pos1 == 0 and posn == n - 1:
            return 0
        
        # Calculate moves to bring 1 to the start
        moves1 = pos1
        
        # Calculate moves to bring n to the end
        movesn = (n - 1) - posn
        
        # If 1 is to the right of n, we need one less move
        if pos1 > posn:
            return moves1 + movesn - 1
        else:
            return moves1 + movesn