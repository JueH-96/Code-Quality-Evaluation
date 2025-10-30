class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        """
        AzeroGPT solution uses a different approach from the provided reasoning process.
        It finds the positions of 1 and n (where n is the last number in the permutation),
        and calculates the minimum number of adjacent swaps needed to move 1 to the start
        and n to the end, adjusting for the case where 1 and n are on the same side.
        """
        n = len(nums)
        left, right = nums.index(1), nums.index(n)
        moves = left
        moves += n - 1 - right
        if left > right:
            moves -= 1
        return moves