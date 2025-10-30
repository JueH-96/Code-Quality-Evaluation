class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Child 1 collects the main diagonal (fixed path)
        child1_sum = sum(fruits[i][i] for i in range(n))
        
        # For n=2, it's a special case where all fruits are collected
        if n == 2:
            return sum(sum(row) for row in fruits)
        
        # DP for children 2 and 3
        # dp[steps][col2][row3] represents max fruits collected by children 2 and 3
        # after 'steps' moves where:
        # - Child 2 is at position (steps, col2) 
        # - Child 3 is at position (row3, steps)
        
        NEG_INF = float('-inf')
        dp = [[[NEG_INF] * n for _ in range(n)] for _ in range(n)]
        
        # Initial state: both children at their starting positions
        # Child 2 at (0, n-1), Child 3 at (n-1, 0)
        dp[0][n-1][n-1] = fruits[0][n-1] + fruits[n-1][0]
        
        # Fill DP table for each step
        for steps in range(n-1):
            for col2 in range(n):
                for row3 in range(n):
                    if dp[steps][col2][row3] == NEG_INF:
                        continue
                    
                    # Try all valid moves for child 2 (from row steps to steps+1)
                    for dc in [-1, 0, 1]:  # Column change
                        new_col2 = col2 + dc
                        if 0 <= new_col2 < n:
                            # Try all valid moves for child 3 (from column steps to steps+1)
                            for dr in [-1, 0, 1]:  # Row change
                                new_row3 = row3 + dr
                                if 0 <= new_row3 < n:
                                    # Calculate new positions
                                    pos2 = (steps + 1, new_col2)
                                    pos3 = (new_row3, steps + 1)
                                    
                                    val = dp[steps][col2][row3]
                                    
                                    # Add fruits, avoiding double counting and diagonal cells
                                    if pos2 == pos3:  # Same cell
                                        if pos2[0] != pos2[1]:  # Not on main diagonal
                                            val += fruits[pos2[0]][pos2[1]]
                                    else:  # Different cells
                                        if pos2[0] != pos2[1]:  # Child 2 not on diagonal
                                            val += fruits[pos2[0]][pos2[1]]
                                        if pos3[0] != pos3[1]:  # Child 3 not on diagonal
                                            val += fruits[pos3[0]][pos3[1]]
                                    
                                    dp[steps+1][new_col2][new_row3] = max(
                                        dp[steps+1][new_col2][new_row3], val
                                    )
        
        # After n-1 steps, both children must be at (n-1, n-1)
        return child1_sum + dp[n-1][n-1][n-1]