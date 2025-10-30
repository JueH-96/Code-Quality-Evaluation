from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0

        # Iterate through the array from the beginning.
        # If nums[i] is 0, we must perform an operation starting at index i.
        # This is because only operations at i-2, i-1, or i can affect nums[i].
        # By processing from left to right, we ensure that elements nums[0]
        # through nums[i-1] are 1 (by applying necessary operations starting
        # at indices 0 through i-1). Thus, if nums[i] is currently 0, we must
        # use the operation starting at index i to make nums[i] a 1 without
        # disturbing the elements to its left.
        # The latest possible starting index for an operation is n-3
        # (affecting n-3, n-2, n-1). We iterate up to n-3.
        for i in range(n - 2):
            if nums[i] == 0:
                operations += 1
                # Apply the flip operation at indices i, i+1, i+2
                # Using XOR 1 is a concise way to flip binary values (0 becomes 1, 1 becomes 0)
                nums[i] ^= 1
                nums[i+1] ^= 1
                nums[i+2] ^= 1

        # After iterating up to index n-3, the elements nums[0] through nums[n-3]
        # should now be 1 according to our greedy strategy.
        # The state of the last two elements, nums[n-2] and nums[n-1], is
        # determined by the operations already applied (specifically, ops
        # starting at indices up to n-3).
        # We check if the last two elements are 1. If not, it's impossible
        # to make all elements 1, because no further operations can affect them
        # (ops at n-2 or n-1 are not allowed) without changing elements before
        # n-2 which we have already made 1.
        for i in range(n - 2, n):
            if nums[i] == 0:
                return -1

        # If the loop finishes and the last two elements are also 1,
        # it means all elements are 1, and we return the total operations count.
        return operations