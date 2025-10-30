class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the positions of 1 and n
        idx1 = nums.index(1)
        idxn = nums.index(n)
        
        # Calculate the cost to bring 1 to front and n to end
        # If n is to the left of 1, we subtract one because
        # when 1 passes n, it effectively reduces one swap operation.
        moves = idx1 + (n - 1 - idxn)
        if idxn < idx1:
            moves -= 1
        
        return moves