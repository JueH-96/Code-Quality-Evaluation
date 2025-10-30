class Solution:
    def maxOperations(self, s: str) -> int:
        """
        The idea is to look at the string as alternating blocks of 1-s and 0-s.
        Every time a block of 0-s starts (i.e. the current character is ‘0’
        and the previous one is ‘1’) all 1-s that have already appeared will
        have to cross that whole block sooner or later.  They will do it
        one-by-one, so the block will add exactly  `number_of_1s_so_far`
        operations – independent of how long the block itself is.

        We therefore walk through the string, count how many 1-s we have seen
        so far and, whenever we meet the first 0 of a block, add that counter
        to the answer.
        """

        ones_so_far = 0      # how many '1's have been seen up to current position
        operations = 0

        for i, ch in enumerate(s):
            if ch == '1':
                ones_so_far += 1
            else:  # ch == '0'
                # first zero of its block ?
                if i > 0 and s[i - 1] == '1':
                    operations += ones_so_far

        return operations