class Solution:
    def maxOperations(self, s: str) -> int:
        """
        We can view the string as alternating blocks of consecutive '0's and '1's.
        A key observation (verified by the examples) is that each block of 1's can
        "jump over" each intervening block of 0's exactly once per 1 in that block,
        regardless of how many consecutive zeros there are (the move skips them all
        in one shot).

        Therefore, we can compute the answer by:
         1) Splitting the string into blocks of consecutive characters.
         2) Keeping track of how many 1's have accumulated so far ("current_ones").
         3) Each time we encounter a block of zeros, we add current_ones to the result
            (each of those 1's jumps over this zero-block in a separate operation).
         4) Each time we encounter a block of ones, we add that block's length to current_ones,
            because now we have more 1's which can jump future zero-blocks.

        The final sum gives the maximum number of operations.
        """

        # Build the blocks of consecutive 0/1
        blocks = []
        n = len(s)
        count = 1
        for i in range(1, n):
            if s[i] == s[i-1]:
                count += 1
            else:
                blocks.append((s[i-1], count))
                count = 1
        # Append the last block
        blocks.append((s[n-1], count))

        answer = 0
        current_ones = 0
        # Traverse the blocks
        for char, length in blocks:
            if char == '1':
                # Add all these 1's to current_ones
                current_ones += length
            else:
                # A block of 0's: each of the existing 1's can jump this block in separate operations
                answer += current_ones

        return answer