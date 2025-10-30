class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        dp = {}

        def solve(row, col, neutralizations):
            if (row, col, neutralizations) in dp:
                return dp[(row, col, neutralizations)]

            if row == m - 1 and col == n - 1:
                return coins[row][col]

            if row >= m or col >= n:
                return float('-inf')

            current_coin = coins[row][col]
            
            ans = float('-inf')
            
            # Move right
            ans = max(ans, current_coin + solve(row, col + 1, neutralizations))
            
            # Move down
            ans = max(ans, current_coin + solve(row + 1, col, neutralizations))
            
            # Neutralize if possible and there is a robber
            if coins[row][col] < 0 and neutralizations > 0:
                ans = max(ans, 0 + solve(row, col + 1, neutralizations - 1))
                ans = max(ans, 0 + solve(row + 1, col, neutralizations - 1))
            
            dp[(row, col, neutralizations)] = ans
            return ans

        return solve(0, 0, 2)