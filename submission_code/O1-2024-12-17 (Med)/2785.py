class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the positions of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # Swaps needed to bring 1 to the front = pos1
        # Swaps needed to bring n to the back = (n-1 - posn)
        # If 1's position is greater than n's position, we subtract 1
        # because one of the swaps that moves 1 past n also moves n to the right.
        if pos1 > posn:
            return pos1 + (n - 1 - posn) - 1
        else:
            return pos1 + (n - 1 - posn)