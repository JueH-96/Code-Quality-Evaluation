class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the indices of 1 and n
        i1 = nums.index(1)
        i2 = nums.index(n)
        
        # Cost to move '1' to the front is i1 swaps
        # Cost to move 'n' to the end is (n - 1 - i2) swaps
        # If 1 is to the right of n (i1 > i2), those moves overlap once, so subtract 1
        if i1 < i2:
            return i1 + (n - 1 - i2)
        else:
            return i1 + (n - 1 - i2) - 1