class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)

        # Mapping of creatures to their defeating counterparts
        defeat = {'F': 'W', 'W': 'E', 'E': 'F'}

        # dp[i][j] will store the number of ways Bob can make i moves and have the j-th creature as the last move
        dp = [[0] * 3 for _ in range n + 1]

        # Initial state: Bob can start with any of the three creatures
        dp[0][0] = dp[0][1] = dp[0][2] = 1

        # Mapping of creatures to indices
        creature_to_index = {'F': 0, 'W': 1, 'E': 2}

        for i in range(1, n + 1):
            current_creature = s[i - 1]
            for j in range(3):
                if j == creature_to_index[current_creature]:
                    continue
                for k in range(3):
                    if k == j or k == creature_to_index[defeat[current_creature]]:
                        continue
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD

        # Summing up all the ways Bob can win
        total_ways = sum(dp[n]) % MOD

        return total_ways