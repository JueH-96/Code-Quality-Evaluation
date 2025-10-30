class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_1 = nums.index(1)
        index_n = nums.index(n)

        swaps = 0

        # Move 1 to the beginning
        swaps += index_1

        # Update the index of n if it was after 1
        if index_n > index_1:
            index_n -= 1

        # Move n to the end
        swaps += (n - 1 - index_n)

        return swaps