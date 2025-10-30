import collections
from typing import List

class Solution:
    """
    Solves the max collected fruits problem using dynamic programming.

    The state dp[k][j2][i3] represents the maximum total fruits collected by all three children up to step k,
    where Child 1 (starting at (0,0)) is at position (k,k), 
    Child 2 (starting at (0, n-1)) is at position (k, j2), and 
    Child 3 (starting at (n-1, 0)) is at position (i3, k).

    Since the computation for step k+1 only depends on the values from step k,
    we can optimize space by using only two layers of the DP table: one for the current step k
    and one for the next step k+1. We use a dictionary `dp` where keys are tuples `(j2, i3)`
    representing the state for the current step `k`, and values are the maximum collected fruits.
    """
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Using defaultdict simplifies initialization. A value of -1 indicates an unreachable state.
        # `dp` stores the DP states for the current step k. Key is tuple (j2, i3).
        dp = collections.defaultdict(lambda: -1) 
        
        # Base case: k=0 (initial positions)
        # Child 1 starts at (0, 0)
        # Child 2 starts at (0, n - 1)
        # Child 3 starts at (n - 1, 0)
        # For n >= 2, these three starting positions are distinct.
        # The fruits at these initial positions are collected at step 0.
        initial_fruits = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        
        # At step k=0, Child 2 is at column j2 = n-1, and Child 3 is at row i3 = n-1.
        # Initialize the DP table for k=0. The only reachable state is (n-1, n-1).
        dp[(n - 1, n - 1)] = initial_fruits

        # Iterate through steps k from 0 to n-2. This covers n-1 moves in total.
        # Each iteration computes the DP states for step k+1 based on step k.
        for k in range(n - 1):
            # `new_dp` will store DP values for the next step k+1
            new_dp = collections.defaultdict(lambda: -1)
            
            # Optimization: Iterate only over states that were reachable in step k.
            # Get the keys (reachable states (j2, i3)) from the current step's dp table.
            reachable_states_k = list(dp.keys()) 

            for state in reachable_states_k:
                 j2, i3 = state  # Current column of Child 2, row of Child 3
                 current_max_fruits = dp[state] # Max fruits collected up to step k ending in this state

                 # If current state was marked unreachable (-1), skip.
                 # This check is technically redundant because we iterate over existing keys, but safe.
                 if current_max_fruits == -1: 
                     continue

                 # Explore possible moves for Child 2 and Child 3 to transition from step k to step k+1
                 
                 # Child 2 moves from (k, j2) to (k+1, j2_new)
                 # Allowed moves change column j by dj2 = -1, 0, 1
                 for dj2 in [-1, 0, 1]:
                    j2_new = j2 + dj2
                    # Check boundary conditions: new column index must be within [0, n-1]
                    if not (0 <= j2_new < n):
                        continue
                    
                    # Child 3 moves from (i3, k) to (i3_new, k+1)
                    # Allowed moves change row i by di3 = -1, 0, 1
                    for di3 in [-1, 0, 1]:
                        i3_new = i3 + di3
                        # Check boundary conditions: new row index must be within [0, n-1]
                        if not (0 <= i3_new < n):
                            continue

                        # Positions of the three children at step k+1:
                        # Child 1 follows a fixed path: P1_new = (k + 1, k + 1)
                        # Child 2 new position: P2_new = (k + 1, j2_new)
                        # Child 3 new position: P3_new = (i3_new, k + 1)
                        
                        # Calculate the fruits collected AT step k+1.
                        # Use a set to automatically handle overlaps: if multiple children land on the same cell,
                        # the fruit is collected only once.
                        visited_cells = set()
                        visited_cells.add((k + 1, k + 1)) # Child 1's cell
                        visited_cells.add((k + 1, j2_new)) # Child 2's cell
                        visited_cells.add((i3_new, k + 1)) # Child 3's cell
                        
                        fruits_step_kp1 = 0
                        for r, c in visited_cells:
                             # Ensure coordinates are valid before accessing fruits grid.
                             # This check should pass due to prior boundary checks but is included for safety.
                             if 0 <= r < n and 0 <= c < n:
                                 fruits_step_kp1 += fruits[r][c]
                            
                        # Calculate the total fruits collected up to step k+1 for this path.
                        new_total_fruits = current_max_fruits + fruits_step_kp1
                        
                        # Update the DP table for the next step (k+1) state (j2_new, i3_new).
                        # Multiple paths (from different states at step k) might lead to the same state at step k+1.
                        # We store the maximum fruits collected among all paths reaching this state.
                        new_state = (j2_new, i3_new)
                        new_dp[new_state] = max(new_dp[new_state], new_total_fruits)

            # After processing all reachable states at step k, update `dp` to the computed `new_dp` for step k+1.
            dp = new_dp 

        # After n-1 steps (loop completes for k=n-2), all children must arrive at the target cell (n-1, n-1).
        # Child 1 reaches (n-1, n-1).
        # Child 2 reaches row n-1. Its final column must be n-1. So final state has j2 = n-1.
        # Child 3 reaches column n-1. Its final row must be n-1. So final state has i3 = n-1.
        # The final state we are interested in is (j2=n-1, i3=n-1).
        final_state = (n - 1, n - 1)
        result = dp[final_state]
        
        # If the target state (n-1, n-1) was unreachable, dp[final_state] would be -1. 
        # Return 0 in this case. Based on the problem rules and n>=2, a path should always exist.
        return result if result != -1 else 0