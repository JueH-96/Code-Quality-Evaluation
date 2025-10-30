from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        """
        Calculates the maximum XOR of a strong pair from a list of integers.

        A pair of integers (x, y) is defined as a "strong pair" if it satisfies
        the condition: |x - y| <= min(x, y).

        This method uses a brute-force approach to check all possible pairs
        of numbers from the input list `nums`. Given the small constraints of the
        problem, this O(n^2) solution is efficient enough.
        """
        n = len(nums)
        max_xor = 0

        # Iterate through all possible pairs of numbers (x, y) from the list.
        for i in range(n):
            x = nums[i]
            
            # Start the inner loop from i to avoid redundant checks of pairs,
            # since both the strong pair condition and XOR are symmetric.
            # This covers all unique pairs (i, j) where i <= j.
            for j in range(i, n):
                y = nums[j]

                # Check if the pair (x, y) is a strong pair.
                if abs(x - y) <= min(x, y):
                    # If it is, calculate its bitwise XOR and update the
                    # overall maximum XOR if the current one is larger.
                    max_xor = max(max_xor, x ^ y)

        return max_xor