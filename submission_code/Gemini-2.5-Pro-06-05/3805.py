from itertools import groupby

class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        """
        Calculates the maximum number of active sections ('1's) after at most one trade.
        """
        n = len(s)
        initial_ones = s.count('1')

        # If the string is all '1's or all '0's, no trade is possible.
        # An all-'1' string has no '0' block to convert.
        # An all-'0' string has no '1' block to give away.
        if initial_ones == n or initial_ones == 0:
            return initial_ones

        groups = [(char, len(list(g))) for char, g in groupby(s)]
        m = len(groups)

        # A trade requires a source '1' block surrounded by '0's (a 0-1-0 pattern),
        # which means we need at least 3 groups.
        if m < 3:
            return initial_ones

        # Create an array of '0' block lengths to build prefix/suffix max arrays.
        only_zero_lengths = [g[1] if g[0] == '0' else 0 for g in groups]
        
        # prefix_max_zeros[i] = max length of a '0' block in groups[0...i]
        prefix_max_zeros = [0] * m
        prefix_max_zeros[0] = only_zero_lengths[0]
        for i in range(1, m):
            prefix_max_zeros[i] = max(prefix_max_zeros[i - 1], only_zero_lengths[i])

        # suffix_max_zeros[i] = max length of a '0' block in groups[i...m-1]
        suffix_max_zeros = [0] * m
        suffix_max_zeros[m - 1] = only_zero_lengths[m - 1]
        for i in range(m - 2, -1, -1):
            suffix_max_zeros[i] = max(suffix_max_zeros[i + 1], only_zero_lengths[i])

        # We can choose not to trade, so the minimum gain we'll accept is 0.
        max_gain = 0
        
        # Iterate through all potential source blocks.
        for i in range(1, m - 1):
            # A valid source block is a '1' block surrounded by '0' blocks.
            if groups[i - 1][0] == '0' and groups[i][0] == '1' and groups[i + 1][0] == '0':
                k_source = groups[i][1]
                l_left_zeros = groups[i - 1][1]
                l_right_zeros = groups[i + 1][1]
                
                # Length of the new '0' block formed by the trade.
                len_merged_zeros = l_left_zeros + k_source + l_right_zeros

                # Max length of any '0' block NOT adjacent to the source block.
                max_l_prefix = prefix_max_zeros[i - 2] if i > 1 else 0
                max_l_suffix = suffix_max_zeros[i + 2] if i < m - 2 else 0
                max_l_other_zeros = max(max_l_prefix, max_l_suffix)
                
                # The best destination is the longest '0' block available after the conversion.
                len_destination = max(len_merged_zeros, max_l_other_zeros)
                
                gain = len_destination - k_source
                max_gain = max(max_gain, gain)
                        
        return initial_ones + max_gain