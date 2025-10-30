import bisect
from typing import List

# Use a sufficiently small integer to represent negative infinity.
# Needs to be smaller than any possible score sum. Minimum possible score sum is 1.
# Maximum possible score sum is roughly 4 * 10^9.
# A value like -10^18 is safe. Adding a small constant like 7 avoids edge cases with exact large negative numbers if they were possible.
NEG_INF = -1 * (10**18 + 7) 

class Solution:
    
    def _better(self, state1, state2):
        """
        Compares two states (score, list_of_indices) and returns the better one.
        A state is considered "better" if it has a higher score. If scores are equal,
        the state with the lexicographically smaller list of indices is considered better.
        Handles NEG_INF scores, which represent unreachable states.
        
        Args:
            state1: Tuple (score1, list1)
            state2: Tuple (score2, list2)
            
        Returns:
            The better state tuple.
        """
        score1, list1 = state1
        score2, list2 = state2

        # Check if states are unreachable (score is NEG_INF)
        is_inf1 = score1 == NEG_INF
        is_inf2 = score2 == NEG_INF

        if is_inf1 and is_inf2:
            # If both states are unreachable, it doesn't matter which one we return.
            # Return state1 for consistency. This path won't lead to an optimal solution anyway.
            return state1 
        elif is_inf1:
            # state1 is unreachable, state2 is reachable. Return state2.
            return state2 
        elif is_inf2:
            # state2 is unreachable, state1 is reachable. Return state1.
            return state1
        
        # If both scores are valid (finite and non-negative)
        if score1 > score2:
            # state1 has a higher score, it's better.
            return state1
        elif score2 > score1:
            # state2 has a higher score, it's better.
            return state2
        else: # Scores are equal, use lexicographical comparison for tie-breaking
            # Python's list comparison directly implements lexicographical order.
            # If list1 is lexicographically smaller than list2, it's better.
            if list1 < list2:
                return state1
            else: # list2 is lexicographically smaller or equal to list1. Return state2.
                  # If lists are identical, returning state2 is fine.
                return state2

    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        """
        Finds a selection of up to 4 non-overlapping intervals that maximizes the total weight.
        Among selections with the maximum possible total weight, it returns the one whose
        indices form the lexicographically smallest list.

        Args:
            intervals: A list of lists, where each inner list is [l_i, r_i, weight_i].

        Returns:
            A list of indices (sorted) representing the chosen intervals.
        """
        
        N = len(intervals)
        # If there are no intervals, return an empty list.
        if N == 0:
            return []

        # Augment each interval with its original 0-based index.
        # This is needed to reconstruct the final list of indices.
        augmented_intervals = []
        for i in range(N):
            augmented_intervals.append(intervals[i] + [i]) # Format: [l, r, w, original_idx]
            
        # Sort intervals primarily based on their end points `r_i`.
        # This sorting order is crucial for the DP approach.
        sorted_intervals = sorted(augmented_intervals, key=lambda x: x[1])

        # Extract end points into a separate list for efficient binary search lookup.
        endpoints = [interval[1] for interval in sorted_intervals]

        K = 4 # The maximum number of non-overlapping intervals we can choose.

        # Initialize the DP table. dp[i][k] will store a tuple: (max_score, indices_list).
        # The dimensions are (N+1) rows (representing using first `i` intervals from the sorted list)
        # and (K+1) columns (representing choosing exactly `k` intervals).
        # Initialize all states to (NEG_INF, []), indicating unreachable state initially.
        dp = [[(NEG_INF, []) for _ in range(K + 1)] for _ in range(N + 1)]

        # Base case: Choosing 0 intervals always results in a score of 0 and an empty list of indices.
        # This is valid regardless of how many intervals (`i`) are considered.
        for i in range(N + 1):
            dp[i][0] = (0, [])

        # Fill the DP table iteratively.
        for i in range(1, N + 1):
            # Get the details of the i-th interval in the sorted list (which has index i-1).
            l, r, w, original_idx = sorted_intervals[i-1]

            # Find the index `p`. `p` is the count of intervals among the first `i-1` sorted intervals
            # (indices 0 to i-2) that end strictly before the start `l` of the current interval.
            # We use binary search (`bisect_right`) on the `endpoints` list for this.
            # `bisect_right(endpoints, V, hi=idx)` searches in `endpoints[0:idx]`.
            # We search for `l-1` in `endpoints[0:i-1]`. `hi=i-1` restricts search to indices 0 to i-2.
            p = bisect.bisect_right(endpoints, l - 1, hi=i-1) 
            
            # Iterate through the possible number of chosen intervals, k, from 1 to K.
            for k in range(1, K + 1):
                # Option 1: Exclude the current interval (interval at index i-1 in sorted list).
                # The best state is inherited from the state considering i-1 intervals and choosing k.
                state1 = dp[i-1][k]

                # Option 2: Include the current interval (interval at index i-1).
                # This is only possible if we can find a valid preceding state.
                state2 = (NEG_INF, []) # Initialize state2 as unreachable.
                
                # We need to choose k-1 intervals from the first p intervals.
                # Check if k-1 is a valid number of intervals (i.e., >= 0).
                # Access the best state found for dp[p][k-1].
                prev_state = dp[p][k-1]
                prev_score, prev_indices = prev_state

                # Check if the previous state `dp[p][k-1]` is reachable (score is not NEG_INF).
                if prev_score != NEG_INF:
                    # Calculate the score for this option.
                    current_score = prev_score + w
                    # Construct the list of indices for this option.
                    # Important: Create a *copy* of `prev_indices` list before modification.
                    current_indices = list(prev_indices) 
                    current_indices.append(original_idx)
                    # Sort the indices list to maintain canonical form for lexicographical comparison.
                    current_indices.sort()
                    # Update state2 with the calculated score and indices list.
                    state2 = (current_score, current_indices)
                
                # Update dp[i][k] by choosing the better state between state1 (exclude) and state2 (include).
                dp[i][k] = self._better(state1, state2)

        # After filling the DP table, find the overall best state.
        # The best state could be achieved by choosing k=0, 1, ..., K intervals.
        # We examine the last row dp[N], which considers all N intervals.
        # Initialize `best_overall_state` with the state for k=0 (score 0, empty list), which is always valid.
        best_overall_state = dp[N][0] 

        # Compare `best_overall_state` with states for k=1 to K.
        for k in range(1, K + 1):
            best_overall_state = self._better(best_overall_state, dp[N][k])

        # The final answer is the list of indices from the best overall state identified.
        return best_overall_state[1]