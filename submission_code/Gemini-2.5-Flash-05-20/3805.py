class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        initial_ones = s.count('1')

        # Augment 's' with '1's at both ends to form 't'.
        # This helps correctly identify blocks surrounded by '0's or '1's,
        # especially near the original string's boundaries.
        # Example: '0100' becomes '101001'.
        t = '1' + s + '1'

        # Parse 't' into a list of alternating block lengths.
        # block_lengths[0] will be the length of the first '1' block (from augmented '1').
        # block_lengths[1] will be the length of the first '0' block.
        # block_lengths[2] will be the length of the second '1' block, and so on.
        # So, lengths at even indices are for '1' blocks, and odd indices for '0' blocks.
        block_lengths = []
        if not t: # Handle empty string case (though constraints say n >= 1)
            return 0
            
        current_char = t[0]
        current_length = 0
        for char in t:
            if char == current_char:
                current_length += 1
            else:
                block_lengths.append(current_length)
                current_char = char
                current_length = 1
        block_lengths.append(current_length) # Add the last block's length

        max_gain = 0

        # We are looking for a pattern: '0' block, '1' block, '0' block
        # in the sequence of block_lengths.
        # In block_lengths: L_0_left (odd index i), L_1_center (index i+1), L_0_right (index i+2)
        # The iteration for 'i' starts from 1 (first '0' block) and increments by 2
        # to ensure it always points to an '0' block.
        # We need at least block_lengths[i], block_lengths[i+1], block_lengths[i+2]
        # so 'i + 2' must be within bounds.
        for i in range(1, len(block_lengths) - 2, 2):
            # block_lengths[i] is the length of the '0' block to the left (L_0_left).
            # block_lengths[i+1] is the length of the '1' block in the center (L_1_center).
            # block_lengths[i+2] is the length of the '0' block to the right (L_0_right).
            
            # As derived, the net gain in '1's for this trade is L_0_left + L_0_right.
            current_gain = block_lengths[i] + block_lengths[i+2]
            max_gain = max(max_gain, current_gain)
            
        # The maximum number of active sections is the initial count of '1's
        # plus the maximum gain achievable from one optimal trade.
        # If no valid trade is possible (max_gain remains 0), the count doesn't change.
        return initial_ones + max_gain