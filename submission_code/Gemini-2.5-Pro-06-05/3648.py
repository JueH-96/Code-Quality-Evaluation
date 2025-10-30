import collections

class Solution:
    def maxCollectedFruits(self, fruits: list[list[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0

        # dp[j][i] stores max fruits collected at step k-1, with child 2 at (k-1, j) and child 3 at (i, k-1)
        dp = [[-1] * n for _ in range(n)]

        # Base case: k=0
        # Child 1 at (0,0), Child 2 at (0, n-1), Child 3 at (n-1, 0).
        # These positions are distinct for n>=2.
        initial_fruits = fruits[0][0] + fruits[0][n - 1] + fruits[n - 1][0]
        dp[n - 1][n - 1] = initial_fruits

        for k in range(1, n):
            new_dp = [[-1] * n for _ in range(n)]
            
            # Iterate over all possible positions for Child 2 (j2) and Child 3 (i3) at step k.
            # The column for child 2 is j2, the row for child 3 is i3.
            for j2 in range(n):
                for i3 in range(n):
                    
                    # Find the maximum value from the previous DP table (at step k-1)
                    # that can transition to the current state.
                    max_prev_dp = -1
                    # Child 2 moves from (k-1, j2_prev) to (k, j2) -> j2_prev in {j2-1, j2, j2+1}
                    # Child 3 moves from (i3_prev, k-1) to (i3, k) -> i3_prev in {i3-1, i3, i3+1}
                    for dj in range(-1, 2):
                        for di in range(-1, 2):
                            j2_prev, i3_prev = j2 + dj, i3 + di
                            
                            if 0 <= j2_prev < n and 0 <= i3_prev < n:
                                if dp[j2_prev][i3_prev] != -1:
                                    max_prev_dp = max(max_prev_dp, dp[j2_prev][i3_prev])

                    if max_prev_dp == -1:
                        continue

                    # Calculate fruits collected at the current step k.
                    # Positions: Child 1 at (k,k), Child 2 at (k,j2), Child 3 at (i3,k)
                    current_fruits = 0
                    pos1 = (k, k)
                    pos2 = (k, j2)
                    pos3 = (i3, k)
                    
                    # Use a set to handle cases where children are in the same room.
                    collected_pos = {pos1, pos2, pos3}
                    for r, c in collected_pos:
                        current_fruits += fruits[r][c]
                    
                    new_dp[j2][i3] = max_prev_dp + current_fruits
            
            dp = new_dp
            
        # After n-1 moves, all children must be at the destination (n-1, n-1).
        # This corresponds to k=n-1, j2=n-1, and i3=n-1 in our DP state.
        final_value = dp[n - 1][n - 1]
        return final_value if final_value != -1 else 0