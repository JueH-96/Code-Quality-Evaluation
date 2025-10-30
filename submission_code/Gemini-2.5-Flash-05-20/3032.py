import math
from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)

        # Determine MAX_J, the maximum power of 2 for binary lifting.
        # We need to consider k+1 steps (terms). k can be up to 10^10.
        # log2(10^10 + 1) is approximately 33.219.
        # So, we need to be able to make jumps corresponding to powers of 2 up to 2^33.
        # A MAX_J value of 34 means indices from 0 to 33, which covers 2^0 to 2^33.
        # This is sufficient to represent any number up to 2^34 - 1 (approx 1.7 * 10^10).
        MAX_J = 34 

        # next_player[j][p]: Stores the ID of the player who receives the ball
        # after 2^j passes, starting from player p.
        next_player = [[0] * n for _ in range(MAX_J)]
        
        # sum_val[j][p]: Stores the sum of player IDs for the sequence of 2^j terms,
        # starting with player p and including the next 2^j - 1 receivers.
        # Formally: p + receiver[p] + receiver[receiver[p]] + ... + receiver^(2^j - 1)[p].
        sum_val = [[0] * n for _ in range(MAX_J)]

        # Base case for binary lifting: j = 0 (2^0 = 1 step)
        # For j=0, next_player[0][p_id] is simply receiver[p_id].
        # sum_val[0][p_id] is the sum of 1 term, which is the player p_id itself.
        for p_id in range(n):
            next_player[0][p_id] = receiver[p_id]
            sum_val[0][p_id] = p_id

        # Precomputation for j > 0
        # Build the tables for 2^j steps based on previously computed 2^(j-1) steps.
        for j in range(1, MAX_J):
            for p_id in range(n):
                # To find the player after 2^j steps from p_id:
                # 1. Find the player after 2^(j-1) steps from p_id.
                intermediate_player = next_player[j-1][p_id]
                # 2. From that intermediate player, take another 2^(j-1) steps.
                next_player[j][p_id] = next_player[j-1][intermediate_player]

                # To find the sum of 2^j terms starting from p_id:
                # 1. Sum the first 2^(j-1) terms (starting from p_id).
                # 2. Sum the next 2^(j-1) terms (these start from intermediate_player).
                # Add these two sums together.
                sum_val[j][p_id] = sum_val[j-1][p_id] + sum_val[j-1][intermediate_player]

        max_f_value = 0

        # For each possible starting player x (from 0 to n-1), calculate f(x)
        # and update the overall maximum function value found.
        for x in range(n):
            current_f_x_sum = 0
            current_player = x
            
            # The function f(x) requires summing x and the players from k passes.
            # This amounts to summing a total of (k + 1) player IDs.
            # E.g., for k=0, sum x (1 term). For k=1, sum x + receiver[x] (2 terms).
            remaining_steps_to_sum = k + 1 

            # Use the binary representation of remaining_steps_to_sum to make jumps.
            # Iterate from the highest power of 2 (MAX_J - 1) down to 0.
            for j in range(MAX_J - 1, -1, -1):
                # If the j-th bit of remaining_steps_to_sum is set (i.e., this power of 2
                # contributes to the total number of steps), then we include this block.
                if (remaining_steps_to_sum >> j) & 1:
                    # Add the sum of IDs for these 2^j steps starting from current_player.
                    current_f_x_sum += sum_val[j][current_player]
                    
                    # Advance the current_player to the player who receives the ball
                    # after these 2^j passes.
                    current_player = next_player[j][current_player]
            
            # Update the maximum function value found so far.
            max_f_value = max(max_f_value, current_f_x_sum)
        
        return max_f_value