class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the positions (indices) of 1 and n
        i1 = nums.index(1)
        i2 = nums.index(n)
        
        # Number of swaps needed to bring 1 to the front is i1
        # Number of swaps needed to bring n to the end is (n - 1 - i2)
        # If i1 > i2, we reduce by 1 because moving 1 left can overlap with moving n right.
        swaps = i1 + (n - 1 - i2)
        if i1 > i2:
            swaps -= 1
        return swaps