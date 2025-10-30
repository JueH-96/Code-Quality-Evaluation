import itertools

class Solution:
    """
    Solves the problem of finding the maximum number of active sections ('1's)
    after at most one trade operation on a binary string s.
    A trade involves two steps performed sequentially:
    1. Convert a contiguous block of '1's that is surrounded by '0's to all '0's.
    2. Afterward, convert a contiguous block of '0's that is surrounded by '1's (in the modified string) to all '1's.
    
    The string s is treated as if it is augmented with '1's at both ends, forming t = '1' + s + '1'. 
    The operations are performed on t. The final count of active sections is taken from the part of t 
    corresponding to the original s. The augmented '1's do not contribute to the final count.
    
    The goal is to maximize the number of '1's in the final string's section corresponding to s.
    We can choose to perform at most one trade. If no trade is performed, the count remains the initial count.
    If a trade is performed, we must select the pair of operations (step 1 followed by step 2)
    that results in the maximum final count.
    """
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        
        n = len(s)
        # Calculate the initial number of active sections ('1's) in the original string s.
        initial_count1 = s.count('1')
        
        # Augment the string s with '1's at both ends to handle boundary conditions easily.
        t = '1' + s + '1'
        
        # Parse the augmented string t into blocks of consecutive identical characters.
        # Each block is represented as a tuple (character, length).
        # Example: "110001001" -> [('1', 2), ('0', 3), ('1', 1), ('0', 2), ('1', 1)]
        blocks = []
        # Use itertools.groupby for efficient block parsing.
        for char, group in itertools.groupby(t):
            blocks.append((char, len(list(group))))
        
        # m is the total number of blocks in the augmented string t.
        m = len(blocks)
        
        # A valid trade requires performing Step 1, which needs a '1' block surrounded by '0's.
        # In the block list representation, this corresponds to a '1' block at an even index `idx` (0-based)
        # such that `0 < idx < m-1`. This implies `idx-1` and `idx+1` exist and are '0' blocks.
        # The minimum structure allowing this is '1' '0' '1' '0' '1', which has 5 blocks (m=5).
        # Example: t = 10101 -> blocks = [('1',1), ('0',1), ('1',1), ('0',1), ('1',1)], m=5. 
        # The '1' block at index 2 is eligible for Step 1.
        # If m < 5, no '1' block is surrounded by '0's, so no trade is possible.
        if m < 5:
            return initial_count1

        # Extract lengths of all '0' blocks from the `blocks` list.
        # '0' blocks occur at odd indices (1, 3, 5, ...) in the 0-based `blocks` list.
        # Store these lengths in list Z. LZ is the count of '0' blocks.
        Z = [blocks[j][1] for j in range(1, m, 2)] 
        LZ = len(Z)

        # If LZ is 0, it implies there are no '0' blocks. This case is covered by m<5 check,
        # as `m>=5` implies `LZ >= (m-1)//2 >= (5-1)//2 = 2`.
        # This check is mainly for logical completeness.
        if LZ == 0:
             return initial_count1 # Should not be reached if m >= 5

        # To efficiently find the maximum length of '0' blocks *not* involved in a merge during Step 1,
        # we precompute prefix and suffix maximums of the lengths in Z.
        
        # Compute prefix maximums: prefix_max[i] = max(Z[0]...Z[i])
        prefix_max = [0] * LZ
        prefix_max[0] = Z[0]
        for i in range(1, LZ):
            prefix_max[i] = max(prefix_max[i-1], Z[i])
        
        # Compute suffix maximums: suffix_max[i] = max(Z[i]...Z[LZ-1])
        suffix_max = [0] * LZ
        suffix_max[LZ-1] = Z[LZ-1]
        for i in range(LZ-2, -1, -1):
            suffix_max[i] = max(suffix_max[i+1], Z[i])
            
        # Initialize the maximum possible final count with the initial count (representing the option of not trading).
        max_final_count = initial_count1
        
        # Iterate through all possible '1' blocks that can be chosen for Step 1 of the trade.
        # These are '1' blocks at even indices `idx` from 2 to `m-3` (inclusive).
        # Such an index `idx` guarantees blocks[idx] is '1', and blocks[idx-1] and blocks[idx+1] are '0'.
        for idx in range(2, m - 1, 2):
            # k is the length of the '1' block chosen for Step 1. Converting this block to '0's decreases the count by k.
            k = blocks[idx][1] 
            
            # After Step 1, the chosen '1' block (now '0's) merges with its adjacent '0' blocks (at idx-1 and idx+1).
            # Calculate the length of this newly formed large '0' block.
            L_merged = blocks[idx-1][1] + k + blocks[idx+1][1]
            
            # Now consider Step 2: convert a '0' block surrounded by '1's to '1's.
            # We want to choose the '0' block with the maximum length `l` to maximize the final count.
            # The candidates for this '0' block are:
            # 1. The merged '0' block of length L_merged. This block is guaranteed to be surrounded by '1's
            #    (originally blocks[idx-2] and blocks[idx+2]).
            # 2. Any other original '0' block that was not part of the merge. These are also surrounded by '1's.
            
            # Find the maximum length among the 'other' original '0' blocks using the precomputed prefix/suffix max arrays.
            # `p1` and `p2` are indices in Z corresponding to the '0' blocks at `blocks[idx-1]` and `blocks[idx+1]`.
            p1 = (idx - 1) // 2  # Index in Z for the '0' block at `blocks[idx-1]`
            p2 = (idx + 1) // 2  # Index in Z for the '0' block at `blocks[idx+1]`
            
            # Initialize max length of other '0' blocks. 0 is a safe minimum as lengths >= 1.
            max_other_l = 0 
            
            # Check max length in Z up to index p1-1 (prefix before p1)
            if p1 > 0:
                max_other_l = max(max_other_l, prefix_max[p1-1])
            # Check max length in Z starting from index p2+1 (suffix after p2)
            if p2 < LZ - 1:
                max_other_l = max(max_other_l, suffix_max[p2+1])

            # The maximum possible length `l` for Step 2 is the maximum of L_merged and max_other_l.
            max_l = max(L_merged, max_other_l)
            
            # Calculate the final count of '1's if this trade (defined by '1' block at idx) is performed.
            # The change in count is `+ max_l` (from Step 2) and `- k` (from Step 1).
            current_trade_count = initial_count1 - k + max_l
            
            # Update the overall maximum final count achieved so far.
            max_final_count = max(max_final_count, current_trade_count)
            
        # After checking all possible trades, return the maximum count found.
        # This includes the possibility of not trading if `initial_count1` was the maximum.
        return max_final_count