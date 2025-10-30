from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)

        # dp[c2][r3] stores max fruits collected after 's' steps,
        # with Child 1 at (s, s), Child 2 at (s, c2), Child 3 at (r3, s).
        # We use two layers for DP states: prev_dp (step s-1) and curr_dp (step s).
        # Initialize with a value indicating unreachable states.
        # Since fruits are non-negative, 0 is a possible total fruit, so use negative infinity.
        prev_dp = [[float('-inf')] * n for _ in range(n)]

        # Base case: s = 0. Children are at initial positions.
        # C1 at (0, 0), C2 at (0, n-1), C3 at (n-1, 0).
        # This corresponds to state indices c2 = n-1, r3 = n-1 in our DP table, when s=0.
        initial_rooms = set()
        initial_rooms.add((0, 0))
        initial_rooms.add((0, n - 1))
        initial_rooms.add((n - 1, 0))

        initial_fruits_set = 0
        for r, c in initial_rooms:
            initial_fruits_set += fruits[r][c]

        # The state at s=0 corresponds to C1 at (0,0), C2 at (0, n-1), C3 at (n-1, 0).
        # This fits the pattern (s, s), (s, c2), (r3, s) with s=0, c2=n-1, r3=n-1.
        # So, the base case DP state is prev_dp[n-1][n-1].
        prev_dp[n - 1][n - 1] = initial_fruits_set

        # Iterate through steps s from 1 to n-1
        # After s steps: C1 is at (s, s), Child 2 at (s, c2), Child 3 at (r3, s)
        for s in range(1, n):
            curr_dp = [[float('-inf')] * n for _ in range(n)]

            # Calculate fruits collected at step s for a given state (s, c2, r3)
            # Children are at (s, s), (s, c2), (r3, s) at the end of step s.
            def get_fruits_at_step(s, c2, r3, fruits):
                rooms = set()
                rooms.add((s, s))       # Child 1
                rooms.add((s, c2))      # Child 2
                rooms.add((r3, s))      # Child 3

                total_fruits = 0
                # Coordinates (s,s), (s,c2), (r3,s) are guaranteed to be within [0, n-1] bounds
                # if s, c2, r3 are within [0, n-1].
                # The loops for c2 and r3 below iterate within the valid ranges [c_min_s, c_max_s].
                for r, c in rooms:
                    total_fruits += fruits[r][c]
                return total_fruits

            # Iterate through all potential ending positions (s, c2) and (r3, s) for step s
            # Child 2 is at (s, c2). Column c2 after s steps starting from n-1.
            # Range: max(0, (n - 1) - s) <= c2 <= min(n - 1, (n - 1) + s)
            # Child 3 is at (r3, s). Row r3 after s steps starting from n-1.
            # Range: max(0, (n - 1) - s) <= r3 <= min(n - 1, (n - 1) + s)

            # Valid range for c2 at step s
            c2_min_s = max(0, (n - 1) - s)
            c2_max_s = min(n - 1, (n - 1) + s)

            # Valid range for r3 at step s
            r3_min_s = max(0, (n - 1) - s)
            r3_max_s = min(n - 1, (n - 1) + s)

            for c2 in range(c2_min_s, c2_max_s + 1):
                for r3 in range(r3_min_s, r3_max_s + 1):

                    max_prev_dp = float('-inf')

                    # Iterate through possible previous column c2_prev for Child 2 at step s-1
                    # C2 moved from (s-1, c2_prev) to (s, c2).
                    # c2 = c2_prev + delta_c2, where delta_c2 in {-1, 0, 1}.
                    # c2_prev = c2 - delta_c2
                    for delta_c2 in [-1, 0, 1]:
                        c2_prev = c2 - delta_c2
                        # Check bounds for c2_prev.
                        if 0 <= c2_prev < n:

                            # Iterate through possible previous row r3_prev for Child 3 at step s-1
                            # C3 moved from (r3_prev, s-1) to (r3, s).
                            # r3 = r3_prev + delta_r3, where delta_r3 in {-1, 0, 1}.
                            # r3_prev = r3 - delta_r3
                            for delta_r3 in [-1, 0, 1]:
                                r3_prev = r3 - delta_r3
                                # Check bounds for r3_prev
                                if 0 <= r3_prev < n:
                                    # If the previous state (s-1, c2_prev, r3_prev) was reachable
                                    if prev_dp[c2_prev][r3_prev] != float('-inf'):
                                        max_prev_dp = max(max_prev_dp, prev_dp[c2_prev][r3_prev])

                    # If state (s, c2, r3) is reachable from any previous state
                    if max_prev_dp != float('-inf'):
                         # Fruits collected at the current step s (when arriving at these rooms)
                        current_fruits = get_fruits_at_step(s, c2, r3, fruits)
                        curr_dp[c2][r3] = current_fruits + max_prev_dp

            # Update prev_dp for the next iteration
            prev_dp = curr_dp

        # The final answer is the maximum fruits collected after n-1 steps
        # when all children are at (n-1, n-1).
        # This corresponds to state (s=n-1, c2=n-1, r3=n-1)
        return prev_dp[n - 1][n - 1]