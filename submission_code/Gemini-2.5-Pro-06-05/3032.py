from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        
        # max_bits is the number of bits in k, determining the number of levels for binary lifting.
        max_bits = k.bit_length()

        # up[j][i]: stores the player who has the ball after 2**j passes, starting from player i.
        # path_sum[j][i]: stores the sum of IDs on the path of length 2**j starting from player i (excluding player i's ID).
        up = [[0] * n for _ in range(max_bits)]
        path_sum = [[0] * n for _ in range(max_bits)]

        # Base case for j=0 (path of length 2**0 = 1).
        for i in range(n):
            up[0][i] = receiver[i]
            path_sum[0][i] = receiver[i]

        # Precompute 'up' and 'path_sum' tables using binary lifting.
        # A jump of 2**j is equivalent to two successive jumps of 2**(j-1).
        for j in range(1, max_bits):
            for i in range(n):
                intermediate_player = up[j - 1][i]
                up[j][i] = up[j - 1][intermediate_player]
                path_sum[j][i] = path_sum[j - 1][i] + path_sum[j - 1][intermediate_player]

        max_f_value = 0
        for i in range(n):
            current_player = i
            # The function value f(i) is the sum of the starting player's ID (i) 
            # and the IDs of the k subsequent receivers.
            current_f_value = i
            
            # Use binary lifting to find the sum of the k receivers' IDs.
            # Decompose k into powers of 2 and take jumps accordingly.
            for j in range(max_bits):
                if (k >> j) & 1:
                    # If j-th bit is set, we take a jump of 2**j.
                    current_f_value += path_sum[j][current_player]
                    current_player = up[j][current_player]
            
            max_f_value = max(max_f_value, current_f_value)
            
        return max_f_value