import itertools

class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        
        # Calculate initial number of ones
        initial_ones = s.count('1')
        
        # Parse s into blocks of alternating characters
        # Store as (char, length, start_index_in_s)
        blocks = []
        if n > 0: # Constraint 1 <= n
            current_char = s[0]
            current_len = 0
            start_index = 0
            for i in range(n):
                if s[i] == current_char:
                    current_len += 1
                else:
                    blocks.append((current_char, current_len, start_index))
                    current_char = s[i]
                    current_len = 1
                    start_index = i
            blocks.append((current_char, current_len, start_index))

        # Get zero block lengths in order of appearance and map start index to list index
        zero_blocks_ordered_lens = [] # List of lengths of zero blocks in order of appearance
        zero_start_idx_to_list_idx = {} # Map start_index_in_s of a zero block to its index in zero_blocks_ordered_lens
        
        zero_block_list_idx = 0
        for block in blocks:
            if block[0] == '0':
                zero_blocks_ordered_lens.append(block[1])
                zero_start_idx_to_list_idx[block[2]] = zero_block_list_idx
                zero_block_list_idx += 1

        P = len(zero_blocks_ordered_lens) # Number of zero blocks
        
        # Compute prefix and suffix max for the list of zero block lengths
        # These will help find the max length of zero blocks *excluding* two specified indices efficiently
        pref_max_Z = [0] * P
        suff_max_Z = [0] * P
        if P > 0:
            pref_max_Z[0] = zero_blocks_ordered_lens[0]
            for i in range(1, P):
                pref_max_Z[i] = max(pref_max_Z[i-1], zero_blocks_ordered_lens[i])
            
            suff_max_Z[P-1] = zero_blocks_ordered_lens[P-1]
            for i in range(P-2, -1, -1):
                suff_max_Z[i] = max(suff_max_Z[i+1], zero_blocks_ordered_lens[i])

        # Identify candidate '1' blocks for step 1
        # Candidate '1' blocks are those surrounded by '0' blocks in the blocks list
        # A '1' block at blocks[i_block] is candidate if blocks[i_block-1] and blocks[i_block+1] exist and are '0' blocks.
        candidate_one_blocks_info = [] # Store (k, L_left_block_list_idx, L_right_block_list_idx)
        
        for i_block in range(len(blocks)):
            char, length, start_index = blocks[i_block]
            
            if char == '1':
                # Check if surrounded by zero blocks in the blocks list
                # blocks[i_block-1] and blocks[i_block+1] must exist and be '0' blocks
                if i_block > 0 and blocks[i_block-1][0] == '0' and \
                   i_block < len(blocks) - 1 and blocks[i_block+1][0] == '0':
                   
                   k = length # Length of the '1' block being converted to '0'
                   
                   # Get the start indices in s for the adjacent zero blocks
                   L_left_block_start_idx_s = blocks[i_block-1][2]
                   L_right_block_start_idx_s = blocks[i_block+1][2]
                   
                   # Get their corresponding indices in the zero_blocks_ordered_lens list
                   L_left_block_list_idx = zero_start_idx_to_list_idx[L_left_block_start_idx_s]
                   L_right_block_list_idx = zero_start_idx_to_list_idx[L_right_block_start_idx_s]
                   
                   candidate_one_blocks_info.append((k, L_left_block_list_idx, L_right_block_list_idx))

        # If no candidate '1' blocks for step 1, no trade is possible according to the rules.
        # In this case, the maximum number of active sections is the initial count.
        if not candidate_one_blocks_info:
            return initial_ones

        # Initialize max_final_ones with the original count.
        # This handles the case where performing a trade does not improve the count,
        # or even reduces it, and the optimal choice is to perform no trade (which is allowed - "at most one trade").
        max_final_ones = initial_ones

        # Iterate through each possible step 1 choice (converting a candidate '1' block to '0's)
        for k, L_left_block_list_idx, L_right_block_list_idx in candidate_one_blocks_info:
            # Get the lengths of the original adjacent zero blocks
            L_left_len = zero_blocks_ordered_lens[L_left_block_list_idx]
            L_right_len = zero_blocks_ordered_lens[L_right_block_list_idx]

            # Calculate the length of the new, larger zero block formed by merging
            L_merged = L_left_len + k + L_right_len

            # Find the length of the longest zero block among all *other* original zero blocks.
            # These are the zero blocks in the original string, excluding the two adjacent ones
            # whose indices in zero_blocks_ordered_lens are L_left_block_list_idx and L_right_block_list_idx.
            max_other_zeros = 0
            
            # Max in Z_len_list_ordered before index L_left_block_list_idx
            if L_left_block_list_idx > 0:
                max_other_zeros = max(max_other_zeros, pref_max_Z[L_left_block_list_idx - 1])
            
            # The indices of zero blocks after the right adjacent one (in the ordered list)
            if L_right_block_list_idx < P - 1:
                 max_other_zeros = max(max_other_zeros, suff_max_Z[L_right_block_list_idx + 1])

            # The longest zero block in the modified string s' (after step 1).
            # This is the candidate for conversion to '1's in step 2 to maximize the gain.
            # It's the maximum of the newly merged block and the longest among the others.
            m = max(L_merged, max_other_zeros)

            # Calculate the total number of ones after performing step 1 (loss of k) and step 2 (gain of m)
            current_final_ones = initial_ones - k + m

            # Update the maximum number of ones found so far across all possible trades.
            max_final_ones = max(max_final_ones, current_final_ones)

        # Return the overall maximum number of active sections (ones).
        # This will be either the initial count or the best count after one trade.
        return max_final_ones