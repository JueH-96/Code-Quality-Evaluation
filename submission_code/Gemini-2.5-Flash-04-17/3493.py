class Solution:
    def maxOperations(self, s: str) -> int:
        """
        Calculates the maximum number of operations on a binary string.

        An operation consists of choosing an index i such that s[i] == '1'
        and s[i+1] == '0' (where i + 1 < s.length), and moving s[i] to the
        right past subsequent '0's until it hits a '1' or the end.

        Args:
            s: A binary string.

        Returns:
            The maximum number of operations possible.
        """
        # Initialize variables
        # ones_count: Tracks the number of '1's encountered so far from the beginning of the string,
        # up to the current index being processed by the loop variable i.
        ones_count = 0
        # total_operations: Accumulates the total number of operations.
        total_operations = 0

        # Iterate through the string from left to right.
        # We need to consider pairs of adjacent characters s[i] and s[i+1] to find "10" patterns.
        # The condition for an operation is s[i] == '1' and s[i+1] == '0'.
        # The index i must satisfy i + 1 < s.length, which means i < s.length - 1.
        # So, the loop should iterate i from 0 up to s.length - 2.
        for i in range(len(s) - 1):
            # Check the character at the current index i.
            # If it's '1', it increases the count of '1's encountered so far
            # (up to and including the current index i).
            if s[i] == '1':
                ones_count += 1
            
            # Check if the current character s[i] is '1' AND the next character s[i+1] is '0'.
            # This specific adjacency s[i]s[i+1] forms the "10" pattern required for an operation
            # starting at index i.
            if s[i] == '1' and s[i+1] == '0':
                # When a "10" pattern is found at index i, it allows the '1' at s[i] to be moved.
                # The crucial observation, supported by examples, is that the maximum number
                # of total operations is achieved by summing the number of '1's encountered
                # up to index i (inclusive) whenever such a "10" pattern is found *in the original string*.
                # `ones_count` at this point represents the number of '1's in s[0...i].
                # These are the '1's that are positioned at or before the '1' initiating this operation.
                # Adding this count to the total seems to capture the maximum potential for operations
                # enabled by this specific "10" pattern's existence in the initial string structure.
                total_operations += ones_count
        
        # After the loop finishes, total_operations holds the accumulated maximum count.
        return total_operations