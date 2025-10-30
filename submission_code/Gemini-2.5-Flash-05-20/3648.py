import collections
from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)

        # dp_prev[(c2, r3)] stores the maximum fruits collected for the previous step (k-1)
        # dp_curr[(c2, r3)] stores the maximum fruits collected for the current step (k)
        
        # Initialize dp_prev with a default value of -float('inf') for unreachable states
        dp_prev = collections.defaultdict(lambda: -float('inf'))

        # Base case: k=0 (initial state)
        # C1: (0,0), C2: (0, n-1), C3: (n-1, 0)
        
        initial_fruits = fruits[0][0]
        # For n >= 2, the three starting positions (0,0), (0,n-1), (n-1,0) are always distinct.
        # So we can simply sum their fruits without overlap checks.
        initial_fruits += fruits[0][n-1]
        initial_fruits += fruits[n-1][0]
        
        # Store the initial state's fruits in dp_prev
        # The key (n-1, n-1) represents c2=n-1 and r3=n-1 for k=0,
        # which are the correct column for C2 and row for C3 at their starting positions.
        dp_prev[(n-1, n-1)] = initial_fruits

        # Iterate k from 1 to n-1 (representing the current number of moves made)
        for k in range(1, n):
            dp_curr = collections.defaultdict(lambda: -float('inf'))
            
            # C1's current position is fixed at (k, k)
            c1_r, c1_c = k, k
            
            # Iterate over all reachable previous states (prev_c2, prev_r3)
            for (prev_c2, prev_r3), prev_max_fruits in dp_prev.items():
                # Skip states that were not reachable in the previous step
                if prev_max_fruits == -float('inf'):
                    continue

                # C2 moves: (i+1, j-1), (i+1, j), (i+1, j+1)
                # Possible changes in c2 are -1, 0, or +1
                c2_delta_moves = [-1, 0, 1] 

                # C3 moves: (i-1, j+1), (i, j+1), (i+1, j+1)
                # Possible changes in r3 are -1, 0, or +1
                r3_delta_moves = [-1, 0, 1] 

                # Explore all 9 possible combinations of moves for C2 and C3
                for dc2 in c2_delta_moves:
                    c2 = prev_c2 + dc2
                    # Ensure C2's new column is within grid bounds
                    if not (0 <= c2 < n):
                        continue
                    
                    for dr3 in r3_delta_moves:
                        r3 = prev_r3 + dr3
                        # Ensure C3's new row is within grid bounds
                        if not (0 <= r3 < n):
                            continue
                        
                        # Calculate fruits collected at current step k
                        current_fruits_at_step_k = 0
                        
                        # Add fruits from C1's position (k,k)
                        current_fruits_at_step_k += fruits[c1_r][c1_c]

                        # Define positions for C2 and C3
                        c2_r, c2_c = k, c2
                        c3_r, c3_c = r3, k

                        # Add fruits from C2's position, if not overlapping with C1
                        if (c2_r, c2_c) != (c1_r, c1_c):
                            current_fruits_at_step_k += fruits[c2_r][c2_c]
                        
                        # Add fruits from C3's position, if not overlapping with C1 or C2
                        if (c3_r, c3_c) != (c1_r, c1_c) and (c3_r, c3_c) != (c2_r, c2_c):
                            current_fruits_at_step_k += fruits[c3_r][c3_c]
                        
                        # Update DP table for the current state (c2, r3) at step k
                        # Take the maximum path that leads to this state
                        dp_curr[(c2, r3)] = max(dp_curr[(c2, r3)], prev_max_fruits + current_fruits_at_step_k)
            
            # After processing all states for step k, update dp_prev for the next iteration
            dp_prev = dp_curr

        # The final answer is the maximum fruits collected when all children reach (n-1, n-1)
        # This state corresponds to k=n-1, c2=n-1, r3=n-1
        final_state_key = (n-1, n-1)
        
        # Return the maximum fruits collected at the final state
        # If the final state is unreachable (e.g., no valid paths exist to it), it will be -inf.
        # For non-negative fruits, 0 is a safe default if no path is found (though it implies an error
        # or special test case if the problem guarantees a path).
        if final_state_key in dp_prev:
            return dp_prev[final_state_key]
        else:
            return 0 # Should not occur given constraints of valid paths to (n-1,n-1)