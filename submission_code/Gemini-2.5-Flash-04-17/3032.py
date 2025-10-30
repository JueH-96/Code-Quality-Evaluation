from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        
        # MAX_POW such that 2^MAX_POW is greater than k.
        # k <= 10^10. log2(10^10) is approximately 33.219.
        # We need powers of 2 up to 2^33.
        # So we need table indices j from 0 to 33. Array size should be 34.
        MAX_POW = 34 

        # next_player_arr[j][i]: the player reached after 2^j passes starting from i.
        next_player_arr = [[0] * n for _ in range(MAX_POW)]

        # sum_of_passes_arr[j][i]: the sum of IDs of the players visited DURING the first 2^j passes,
        # starting from i. This means the sum of players receiver^1[i] + ... + receiver^{2^j}[i].
        sum_of_passes_arr = [[0] * n for _ in range(MAX_POW)]

        # Base case j = 0 (2^0 = 1 pass)
        for i in range(n):
            # Player reached after 1 pass from i is receiver[i].
            next_player_arr[0][i] = receiver[i]
            # Sum of players DURING 1 pass from i is just the receiver: receiver[i].
            sum_of_passes_arr[0][i] = receiver[i]

        # Fill DP table for j = 1 to MAX_POW-1
        for j in range(1, MAX_POW):
            for i in range(n):
                # To find player after 2^j passes from i:
                # Go 2^(j-1) passes from i to reach node u = next_player_arr[j-1][i].
                # Then go another 2^(j-1) passes from u to reach node next_player_arr[j-1][u].
                u = next_player_arr[j-1][i]
                next_player_arr[j][i] = next_player_arr[j-1][u]

                # To find sum of players DURING 2^j passes from i:
                # Sum of players during first 2^(j-1) passes from i = sum_of_passes_arr[j-1][i].
                # Sum of players during next 2^(j-1) passes from i: These passes start
                # from the player reached after the first 2^(j-1) passes from i,
                # which is u = next_player_arr[j-1][i].
                # The sum of players during 2^(j-1) passes starting from u is sum_of_passes_arr[j-1][u].
                # Total sum = sum_of_passes_arr[j-1][i] + sum_of_passes_arr[j-1][u]
                sum_of_passes_arr[j][i] = sum_of_passes_arr[j-1][i] + sum_of_passes_arr[j-1][u]

        max_f = 0

        # Calculate f(x) for each starting player x from 0 to n-1
        # f(x) = x + (sum of k passes starting from x).
        # The sum of k passes is p_1 + ... + p_k.
        
        for x in range(n):
            current_node = x
            current_sum = x # Start with p_0 = x

            # Now add the sum of k passes. Use binary decomposition of k.
            temp_k = k 
            
            # Iterate through the bits of k from MAX_POW-1 down to 0.
            # If the j-th bit of k is 1, it means k includes a term 2^j.
            # We should add the sum of players during 2^j passes starting from current_node.
            # And update current_node by taking 2^j passes.

            # We need to check bits up to the highest bit possible for k <= 10^10.
            # The highest bit is 33. So we need j from 33 down to 0.
            # The range `range(MAX_POW - 1, -1, -1)` with `MAX_POW = 34` gives `range(33, -1, -1)`.
            # This covers j = 33, 32, ..., 0. Correct.

            for j in range(MAX_POW - 1, -1, -1):
                # If the j-th bit of k is set
                if (temp_k >> j) & 1:
                    # Add the sum of the next 2^j passes starting from current_node.
                    current_sum += sum_of_passes_arr[j][current_node]
                    # Move the current_node forward by 2^j passes.
                    current_node = next_player_arr[j][current_node]

            max_f = max(max_f, current_sum)

        return max_f