import sys
# Setting higher recursion depth is not needed for iterative DP
# sys.setrecursionlimit(2000) 

class Solution:
    def stringCount(self, n: int) -> int:
        """
        Counts the number of strings of length n such that they can be rearranged 
        to contain "leet" as a substring. This condition is equivalent to counting strings
        that contain at least one 'l', at least two 'e's, and at least one 't'.
        This solution uses Dynamic Programming.

        The state of our DP represents the counts of 'l', 'e', 't' encountered so far.
        Since we only care about whether the counts reach certain thresholds (1 for 'l', 2 for 'e', 1 for 't'),
        we can define the DP state based on these thresholds.

        State definition:
        We use a tuple (l_state, e_state, t_state) to represent the state:
          l_state: 0 if no 'l' has been encountered, 1 if at least one 'l' has been encountered.
          e_state: 0 if no 'e' has been encountered, 1 if exactly one 'e' has been encountered, 2 if at least two 'e's have been encountered.
          t_state: 0 if no 't' has been encountered, 1 if at least one 't' has been encountered.

        The DP table `dp[l_state][e_state][t_state]` will store the number of strings of the current length
        that satisfy the conditions specified by the state (l_state, e_state, t_state).

        To optimize space and potentially improve cache locality, we can map the 3D state into a 1D index.
        The state (l, e, t) can be mapped to a unique index `state_idx = l * 6 + e * 2 + t`.
        The dimensions are: l=2, e=3, t=2. The mapping works as follows:
          Index = l * (3 * 2) + e * (2) + t = l * 6 + e * 2 + t
        The total number of states is 2 * 3 * 2 = 12. Indices range from 0 to 11.
        
        Base Case: For a string of length 0 (empty string), the state is (0, 0, 0), corresponding to index 0. So, dp[0] = 1.
        
        Transitions: For each state at length `i`, we consider appending one character to transition to length `i+1`.
        If we append 'l': The l_state becomes 1. The e_state and t_state remain unchanged.
        If we append 'e': The e_state increments, capped at 2. The l_state and t_state remain unchanged.
        If we append 't': The t_state becomes 1. The l_state and e_state remain unchanged.
        If we append any other character (23 choices): The state (l_state, e_state, t_state) remains unchanged.
        
        We iterate `n` times, updating the DP table at each step. The final answer is the count in the state (1, 2, 1), 
        which corresponds to index `1 * 6 + 2 * 2 + 1 = 11`.
        All calculations are performed modulo 10^9 + 7.
        """
        
        M = 1000000007
        
        # Initialize DP table for length 0. Use a flat list/array of size 12.
        # dp[state_idx] corresponds to the count for that state.
        dp = [0] * 12
        
        # Base case: empty string "" corresponds to state (l=0, e=0, t=0).
        # state_idx = 0*6 + 0*2 + 0 = 0
        dp[0] = 1 

        # Iterate n times, simulating building the string character by character
        for _ in range(n):
            # Create a new DP table for the next length based on the current `dp` table.
            new_dp = [0] * 12
            
            # Iterate through all 12 possible states for strings of the current length
            for state_idx in range(12):
                # Get the count of strings currently in this state
                count = dp[state_idx]
                if count == 0: continue # Skip if this state is unreachable

                # Deconstruct state_idx to get the individual l, e, t states
                # l = state_idx // 6
                # e = (state_idx % 6) // 2
                # t = state_idx % 2
                # Using divmod for potentially slight efficiency:
                l, rem = divmod(state_idx, 6)
                e, t = divmod(rem, 2)

                # --- Calculate transitions based on appending one character ---

                # 1. Append 'l'
                # New state: l becomes 1. e and t unchanged.
                new_l = 1 
                new_state_idx_l = new_l * 6 + e * 2 + t
                new_dp[new_state_idx_l] = (new_dp[new_state_idx_l] + count) % M
                        
                # 2. Append 'e'
                # New state: e increments, capped at 2. l and t unchanged.
                new_e = min(e + 1, 2) 
                new_state_idx_e = l * 6 + new_e * 2 + t
                new_dp[new_state_idx_e] = (new_dp[new_state_idx_e] + count) % M
                        
                # 3. Append 't'
                # New state: t becomes 1. l and e unchanged.
                new_t = 1
                new_state_idx_t = l * 6 + e * 2 + new_t
                new_dp[new_state_idx_t] = (new_dp[new_state_idx_t] + count) % M
                        
                # 4. Append any other character (23 choices: alphabet size 26 - 3 specific chars 'l', 'e', 't')
                # State remains the same as current state: state_idx
                # We multiply the count by 23 because there are 23 characters that lead to this transition.
                new_dp[state_idx] = (new_dp[state_idx] + count * 23) % M
                        
            # Update the dp table to the newly computed state counts for the next iteration
            dp = new_dp
            
        # The final answer is the count of strings of length n that satisfy all conditions:
        # At least one 'l' (l=1), at least two 'e's (e=2), and at least one 't' (t=1).
        # This corresponds to state index: 1 * 6 + 2 * 2 + 1 = 6 + 4 + 1 = 11.
        return dp[11]