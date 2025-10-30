class Solution:
    def minimumSteps(self, s: str) -> int:
        """
        Calculates the minimum number of steps (adjacent swaps) required
        to group all black balls ('1') to the right and white balls ('0') to the left.

        The problem asks for the minimum number of adjacent swaps to transform the
        given string `s` into a target configuration where all '0's are on the left
        and all '1's are on the right (e.g., "000111").

        This minimum number of adjacent swaps is equivalent to the number of
        inversions needed to sort the sequence into the target order ('0's before '1's).
        An inversion, in this context, is a pair of indices (i, j) such that i < j,
        but the element at index i should come after the element at index j in the
        target configuration. This happens when s[i] = '1' (black ball) and
        s[j] = '0' (white ball).

        We can efficiently count these inversions by iterating through the string
        from left to right. We maintain a count of the black balls ('1's) encountered
        so far. When we encounter a white ball ('0'), it needs to move past all the
        black balls currently to its left to reach its correct final position.
        The number of swaps required for this specific white ball is equal to the
        number of black balls encountered before it. Adding this count to a running
        total gives the total minimum swaps required.

        Args:
            s: The 0-indexed binary string representing the balls, where '0' is a
               white ball and '1' is a black ball. The length n of the string
               satisfies 1 <= n <= 10^5.

        Returns:
            The minimum number of steps (adjacent swaps) required. The result can be
            large, but Python's arbitrary precision integers handle the potential size.
        """
        n = len(s)
        # If the string has only one ball, no swaps are needed.
        # The constraints state n >= 1, so n=0 is not possible.
        if n <= 1:
            return 0

        total_swaps: int = 0  # Initialize the total number of swaps to 0
        ones_count: int = 0   # Initialize the count of black balls ('1's) seen so far to 0

        # Iterate through the string character by character from left to right
        for char in s:
            if char == '1':
                # If the current character is a black ball ('1'), increment the count
                # of black balls encountered so far.
                ones_count += 1
            else: # char == '0'
                # If the current character is a white ball ('0'), this ball needs
                # to be effectively moved to the left past all the black balls ('1's)
                # that have been encountered previously. The number of swaps
                # required for this move is equal to the current `ones_count`.
                # Add this number to the total swaps.
                total_swaps += ones_count
                
        # After iterating through the entire string, total_swaps will hold the
        # total minimum number of adjacent swaps needed.
        return total_swaps