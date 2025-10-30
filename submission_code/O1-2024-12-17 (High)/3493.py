class Solution:
    def maxOperations(self, s: str) -> int:
        """
        We want to count the maximum number of times we can pick an index i with s[i] == '1' and s[i+1] == '0'
        and "move" that '1' to the right (skipping all consecutive zeros) as a single operation.  

        A neat (and perhaps surprising) way to get the answer—matching all the provided examples—is:

          1. Traverse the string from right to left.
          2. Keep track of how many "blocks of consecutive zeros" we have seen so far. 
             Concretely, each time we encounter a '0' that is NOT directly followed by another '0', 
             we increment the count of zero-blocks.
          3. Each time we see a '1', we add the current number of zero-blocks to our answer.
          
        Why this works for the given examples:
          - A “block of zeros” is a (maximal) run of consecutive '0's in left-to-right order.
            As we scan right-to-left, whenever we see the start of a new block (i.e. s[i]=='0' but s[i+1] != '0'),
            we count one more block.
          - Each '1' will, in effect, be able to “jump over” (in separate operations) all the zero-blocks 
            that lie strictly to its right, and each such jump contributes +1 to the operation count.
          
        This matches the examples in the prompt:
        
          Example 1: s = "1001101"  -->  4
          Example 2: s = "00111"    -->  0
          Also test:   s = "010010" -->  3
          
        and agrees exactly with the procedure described in those examples.
        """

        ans = 0
        zero_blocks = 0
        n = len(s)

        # Scan from right to left
        for i in reversed(range(n)):
            if s[i] == '0':
                # If this zero is NOT directly followed by another zero (or is at the end),
                # then we've found a new block of zeros.
                if i + 1 < n and s[i+1] == '0':
                    # same block, do nothing
                    pass
                else:
                    # new block
                    zero_blocks += 1
            else:
                # s[i] == '1' --> add however many zero-blocks we've seen
                ans += zero_blocks

        return ans