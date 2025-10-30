class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # dp[i][j][k] is a bitmask of reachable end-states 
        # after using i "AA", j "BB", k "AB" blocks.
        # States encoded in 2 bits: 
        #   bit0 -> state 0: no last char (start)
        #   bit1 -> state 1: last run is 'A' of length 1
        #   bit2 -> state 2: last run is 'A' of length 2
        #   bit3 -> state 3: last run is 'B' (we collapse B-length1/2 into one state)
        dp = [[[0] * (z + 1) for _ in range(y + 1)] for __ in range(x + 1)]
        dp[0][0][0] = 1 << 0  # start in state 0
        
        # Precompute masks for which states allow each block:
        mask_AA = (1 << 0) | (1 << 3)        # AA allowed if last_char != 'A'
        mask_BB = (1 << 0) | (1 << 1) | (1 << 2)  # BB allowed if last_char != 'B'
        mask_AB = (1 << 0) | (1 << 1) | (1 << 3)  # AB allowed if not in state2 (A run of length2)
        
        # Next-state bits for each block
        ns_AA = (1 << 2)  # AA ends in 'A' run of length 2
        ns_BB = (1 << 3)  # BB ends in 'B'
        ns_AB = (1 << 3)  # AB ends in 'B'
        
        max_blocks = 0
        for i in range(x + 1):
            for j in range(y + 1):
                for k in range(z + 1):
                    bits = dp[i][j][k]
                    if bits:
                        # update maximum number of blocks used
                        max_blocks = max(max_blocks, i + j + k)
                        # try to add an "AA" block
                        if i < x and (bits & mask_AA):
                            dp[i + 1][j][k] |= ns_AA
                        # try to add a "BB" block
                        if j < y and (bits & mask_BB):
                            dp[i][j + 1][k] |= ns_BB
                        # try to add an "AB" block
                        if k < z and (bits & mask_AB):
                            dp[i][j][k + 1] |= ns_AB
        
        # Each block has length 2
        return max_blocks * 2