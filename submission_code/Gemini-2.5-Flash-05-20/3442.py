import collections

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Step 1: Sort rewardValues and remove duplicates.
        # This is an optimization as duplicate values of `r` do not offer
        # additional unique paths to a sum `x+r` if `x >= r`.
        # The 'val > current_x' condition ensures that we can only add
        # rewards that are strictly greater than the current sum.
        rewards = sorted(list(set(rewardValues)))

        # Step 2: Determine the maximum possible sum.
        # The maximum value for rewardValues[i] is 2000.
        # The maximum possible total reward can be up to (2 * max_reward_value - 1).
        # For example, if max_reward_value is M, and we have M-1, M.
        # We can form (M-1) from previous rewards, then add M: (M-1) + M = 2M - 1.
        # So, for max_reward_value = 2000, max sum is 2 * 2000 - 1 = 3999.
        # The array size should be 4000 to cover indices 0 to 3999.
        MAX_POSSIBLE_SUM = 4000 
        
        # Step 3: Initialize dp_reachable array.
        # dp_reachable[s] is True if sum 's' can be achieved, False otherwise.
        dp_reachable = [False] * MAX_POSSIBLE_SUM
        
        # Step 4: Base case - 0 reward is always achievable (by taking no rewards).
        dp_reachable[0] = True
        
        # Step 5: Keep track of the maximum total reward achieved so far.
        max_total_reward_achieved = 0

        # Step 6: Iterate through each unique and sorted reward value.
        for val in rewards:
            # Iterate through all possible current sums 'current_x'.
            # We can iterate from 0 upwards. If dp_reachable[current_x] is True,
            # and val > current_x, then current_x + val is a new possible sum.
            # This doesn't affect `current_x` in the same iteration because `current_x + val > current_x`.
            # So, we are always working with `dp_reachable` values from the *previous* state
            # (i.e., before adding the current `val`).
            for current_x in range(MAX_POSSIBLE_SUM):
                if dp_reachable[current_x]:
                    # Check the condition: rewardValues[i] must be greater than current total reward x.
                    if val > current_x:
                        new_sum = current_x + val
                        # Ensure new_sum is within array bounds.
                        if new_sum < MAX_POSSIBLE_SUM:
                            dp_reachable[new_sum] = True
                            # Update the maximum reward achieved if this new_sum is higher.
                            max_total_reward_achieved = max(max_total_reward_achieved, new_sum)
        
        # Step 7: The maximum value in dp_reachable (or tracked by max_total_reward_achieved)
        # represents the maximum total reward that can be collected.
        return max_total_reward_achieved