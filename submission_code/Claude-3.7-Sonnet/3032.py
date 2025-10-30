class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_bit = min(35, k.bit_length())  # log2(10^10) ≈ 33, adding buffer
        
        # Initialize tables for binary lifting
        jump_next = [[0] * max_bit for _ in range(n)]
        jump_sum = [[0] * max_bit for _ in range(n)]
        
        # Base case: 2^0 = 1 step
        for i in range(n):
            jump_next[i][0] = receiver[i]
            jump_sum[i][0] = receiver[i]
        
        # Precompute jumps for powers of 2
        for j in range(1, max_bit):
            for i in range(n):
                next_player = jump_next[i][j-1]
                jump_next[i][j] = jump_next[next_player][j-1]
                jump_sum[i][j] = jump_sum[i][j-1] + jump_sum[next_player][j-1]
        
        # Compute f(x) for each starting player and find maximum
        max_f = 0
        for x in range(n):
            curr_player = x
            f_x = x  # Starting with the initial player's ID
            
            # Process each bit of k to simulate exactly k passes
            for j in range(max_bit):
                if (k >> j) & 1:  # If jth bit of k is set
                    f_x += jump_sum[curr_player][j]
                    curr_player = jump_next[curr_player][j]
            
            max_f = max(max_f, f_x)
        
        return max_f