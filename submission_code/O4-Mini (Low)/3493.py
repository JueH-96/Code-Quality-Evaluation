class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        # collect the indices of '1's
        ones = [i for i, ch in enumerate(s) if ch == '1']
        if not ones:
            return 0
        
        ops = 0
        m = len(ones)
        # For each '1', look at the block of zeros immediately after it:
        # - between this '1' and the next '1'
        # - for the very last '1', between it and the end of string
        # If that block has length > 0, then all the ones to the left
        # of that block (inclusive) can each cross it once, contributing
        # (index_in_ones_list + 1) operations.
        for idx, pos in enumerate(ones):
            # determine end of this block
            if idx + 1 < m:
                nxt = ones[idx+1]
            else:
                nxt = n
            zero_block_len = nxt - pos - 1
            if zero_block_len > 0:
                # there are (idx+1) ones to the left (including this one)
                ops += (idx + 1)
        return ops