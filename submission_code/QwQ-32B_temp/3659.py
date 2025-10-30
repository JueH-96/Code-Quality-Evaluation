class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0])
        
        # Initialize previous row's DP array
        prev_dp = [[0] * 16 for _ in range(n)]
        
        # Starting cell (0, 0)
        first_val = grid[0][0]
        prev_dp[0][first_val] = 1
        
        # Fill the first row
        for j in range(1, n):
            val = grid[0][j]
            temp = [0] * 16
            for prev_x in range(16):
                cnt = prev_dp[j-1][prev_x]
                new_x = prev_x ^ val
                temp[new_x] = (temp[new_x] + cnt) % MOD
            # Update prev_dp[j] with temp values
            for x in range(16):
                prev_dp[j][x] = temp[x]
        
        # Process remaining rows
        for i in range(1, m):
            current_row = [[0] * 16 for _ in range(n)]
            for j in range(n):
                val = grid[i][j]
                temp = [0] * 16
                # Contribution from the top (prev_dp[j])
                for prev_x in range(16):
                    cnt = prev_dp[j][prev_x]
                    new_x = prev_x ^ val
                    temp[new_x] = (temp[new_x] + cnt) % MOD
                # Contribution from the left (current_row[j-1] if j > 0)
                if j > 0:
                    for prev_x in range(16):
                        cnt = current_row[j-1][prev_x]
                        new_x = prev_x ^ val
                        temp[new_x] = (temp[new_x] + cnt) % MOD
                # Assign temp to current_row[j]
                for x in range(16):
                    current_row[j][x] = temp[x] % MOD
            # Update prev_dp to current_row for next iteration
            prev_dp = current_row
        
        # The result is the count for the last cell's XOR value k
        return prev_dp[n-1][k] % MOD