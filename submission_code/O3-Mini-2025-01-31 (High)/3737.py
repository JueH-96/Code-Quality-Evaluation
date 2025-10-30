from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # We'll use a pairing idea.
        # Since n is even, we can form M = n//2 pairs where pair i consists of:
        #    left house: index i
        #    right house: index n-1-i
        #
        # The mirror condition “houses equidistant from the ends not painted the same color”
        # means that for each pair i the colors must be different.
        #
        # The adjacent condition (no two adjacent houses painted the same color)
        # in street order gives:
        #   - For the left-half houses (indices 0,1,...,M-1): colors must differ consecutively.
        #   - For the right-half houses (indices n-M, ..., n-2, n-1) the actual order is reversed.
        #     With a simple observation, this is equivalent to requiring that if for each pair i we denote
        #     the color used for the right side as R[i], then in natural pair order,
        #     we need R[i] != R[i+1] (because the inequality "!=" is symmetric).
        #
        # Therefore, for each pair i we choose a state: (L, R) with L,R in {0,1,2} and L != R.
        # There are 6 valid states:
        #    (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        #
        # The cost of a pair i and state (a,b) is defined as:
        #    pair_cost = cost[i][a] + cost[n-1-i][b]
        #
        # In addition, when transitioning from pair i to pair i+1, we must have:
        #    left side:   state[i].L != state[i+1].L
        #    right side:  state[i].R != state[i+1].R
        #
        # Then, our answer is the minimum total cost over pairs 0..M-1.
        
        M = n // 2
        # List all possible valid (left, right) color choices (colors are 0-indexed)
        states = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        num_states = len(states)  # 6 states

        # dp[s] will store the minimum cost up to the current pair with the pair's state = states[s]
        dp = [float('inf')] * num_states

        # For pair 0 (houses 0 and n-1)
        left_index = 0
        right_index = n - 1
        for s, (clr_left, clr_right) in enumerate(states):
            dp[s] = cost[left_index][clr_left] + cost[right_index][clr_right]

        # Process pairs 1 to M-1
        for i in range(1, M):
            new_dp = [float('inf')] * num_states
            left_index = i
            right_index = n - 1 - i
            # current pair cost for each state
            curr_cost = [0] * num_states
            for s, (clr_left, clr_right) in enumerate(states):
                curr_cost[s] = cost[left_index][clr_left] + cost[right_index][clr_right]
            # Transition from previous pair to current pair:
            # We require: previous state's left color != current state's left color,
            # and previous state's right color != current state's right color.
            for curr_state in range(num_states):
                curr_left, curr_right = states[curr_state]
                best_prev = float('inf')
                for prev_state in range(num_states):
                    prev_left, prev_right = states[prev_state]
                    if prev_left != curr_left and prev_right != curr_right:
                        if dp[prev_state] < best_prev:
                            best_prev = dp[prev_state]
                new_dp[curr_state] = best_prev + curr_cost[curr_state]
            dp = new_dp
        
        return min(dp)