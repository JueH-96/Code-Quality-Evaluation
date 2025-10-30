class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)

        # Calculate initial number of '1's
        initial_ones = s.count('1')

        # If no trade is possible or beneficial, this is the result.
        # We will find the max net gain from a trade and add it to this.
        max_net_gain = 0

        # Build blocks list: (start_index, end_index, char)
        # This helps in easily identifying contiguous blocks and their neighbors.
        blocks = []
        if n > 0:
            i = 0
            while i < n:
                start = i
                char = s[i]
                while i < n and s[i] == char:
                    i += 1
                end = i - 1
                blocks.append((start, end, char))

        # Iterate through blocks to find potential Step 1 candidates
        # A '1' block is a Step 1 candidate if it's surrounded by '0' blocks in s.
        # In the blocks list, this means a '1' block at index k (0 < k < len(blocks)-1)
        # such that blocks[k-1] is a '0' block and blocks[k+1] is a '0' block.
        
        for k in range(1, len(blocks) - 1):
            if blocks[k][2] == '1' and blocks[k-1][2] == '0' and blocks[k+1][2] == '0':
                # This blocks[k] is a valid '1' block to zero out (Step 1)
                p, q = blocks[k][0], blocks[k][1] # Start/end indices of the '1' block
                loss = q - p + 1 # Number of '1's lost in Step 1

                # After zeroing out blocks[k], the contiguous block of '0's available
                # for Step 2 spans from the start of the preceding '0' block (blocks[k-1])
                # to the end of the succeeding '0' block (blocks[k+1]) in original indices.
                # This entire range (original s[i...j]) becomes '0's in the modified string s'.
                i = blocks[k-1][0] # Start index of the combined potential '0' block
                j = blocks[k+1][1] # End index of the combined potential '0' block

                gain = j - i + 1 # Length of the combined '0' block (number of '0's turned to '1's in Step 2)

                # Check if this combined '0' block (s'[i...j]) is surrounded by '1's
                # in the augmented string '1' + s' + '1'.
                # This means the character immediately before index i in s' (or augmented '1') is '1'
                # AND the character immediately after index j in s' (or augmented '1') is '1'.

                # The character before index i in the original string s is at index i-1 (if i > 0).
                # In the blocks list, this character is part of blocks[k-2] if k-2 exists.
                # If i == 0, the character before is the augmented '1'.
                is_surrounded_left = (blocks[k-1][0] == 0) or (k - 2 >= 0 and blocks[k-2][2] == '1')

                # The character after index j in the original string s is at index j+1 (if j < n-1).
                # In the blocks list, this character is part of blocks[k+2] if k+2 exists.
                # If j == n - 1, the character after is the augmented '1'.
                is_surrounded_right = (blocks[k+1][1] == n - 1) or (k + 2 < len(blocks) and blocks[k+2][2] == '1')

                # If the combined '0' block is surrounded by '1's in the augmented string,
                # it's a valid candidate for conversion to '1's in Step 2.
                if is_surrounded_left and is_surrounded_right:
                    current_net_gain = gain - loss
                    max_net_gain = max(max_net_gain, current_net_gain)
                    
        # The maximum number of active sections is the initial count
        # plus the maximum net gain found from any single valid trade.
        # If max_net_gain remains 0, it means no beneficial trade was possible or found.
        return initial_ones + max_net_gain