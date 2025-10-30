class Solution:
    def maxOperations(self, s: str) -> int:
        """
        Calculates the maximum number of operations that can be performed on a binary string.

        The key insight is that the maximum number of operations is achieved by summing,
        for each '1', the number of contiguous blocks of '0's to its right.
        This can be calculated efficiently with a single pass from right to left.
        """
        n = len(s)
        if n <= 1:
            return 0

        total_ops = 0
        # This counts the number of '0' blocks found to the right of the current position.
        zero_groups_count = 0

        # Iterate from right to left.
        for i in range(n - 1, -1, -1):
            if s[i] == '1':
                # This '1' must jump over all '0' blocks to its right.
                # The number of such operations for this '1' is the number of
                # zero groups we have encountered so far.
                total_ops += zero_groups_count
            else:  # s[i] == '0'
                # We check if this '0' is the rightmost '0' of its contiguous block.
                # This is true if it's at the end of the string or followed by a '1'.
                # This ensures we count each block of zeros only once.
                if i < n - 1 and s[i + 1] == '1':
                    zero_groups_count += 1
        
        return total_ops