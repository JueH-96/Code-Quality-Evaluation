class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        # Find the current index of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(len(nums))
        
        # Moving '1' to the front needs pos1 swaps.
        # Moving 'n' to the end needs (len(nums) - 1 - posn) swaps.
        # However, if '1' is to the right of 'n' (i.e., pos1 > posn),
        # then while shifting '1' to the front, 'n' is effectively shifted one position to the right,
        # so we subtract one swap from the total.
        if pos1 > posn:
            return pos1 + (len(nums) - 1 - posn) - 1
        else:
            return pos1 + (len(nums) - 1 - posn)