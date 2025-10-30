import itertools
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        """
        Calculates the maximum number that can be formed by concatenating the binary
        representations of the elements of nums in some order.

        Args:
          nums: A list of 3 integers.

        Returns:
          The maximum possible integer value.
        """

        # Since the input list `nums` has a fixed length of 3, the number of
        # possible orderings (permutations) is small (3! = 6). We can generate
        # all permutations, compute the resulting number for each, and find the maximum.
        
        # The logic is encapsulated in a generator expression passed to max():
        # 1. `itertools.permutations(nums)`: Generates all orderings, e.g., (1, 2, 3), (1, 3, 2), ...
        # 2. `bin(num)[2:]`: For each number, get its binary string without the '0b' prefix.
        # 3. `"".join(...)`: Concatenates the binary strings for a given permutation.
        # 4. `int(..., 2)`: Converts the concatenated binary string into an integer.
        # 5. `max(...)`: Finds the maximum value among all generated integers.
        #
        # Note: We import `itertools` inside the function, a common practice in
        # competitive programming to ensure the code is self-contained.

        return max(
            int("".join(bin(num)[2:] for num in p), 2)
            for p in itertools.permutations(nums)
        )