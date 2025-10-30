class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        def get_score(alice, bob):
            if alice == bob:
                return 0  # tie
            if (alice == 'F' and bob == 'W') or \
               (alice == 'W' and bob == 'E') or \
               (alice == 'E' and bob == 'F'):
                return 1  # Bob wins
            return -1  # Alice wins
        
        creatures = ['F', 'W', 'E']
        # dp[i][last][diff] = number of ways to reach round i,
        # with Bob's last creature being 'last', and score difference being 'diff'
        # Score difference is offset by n to handle negative values
        dp = [[[0] * (2*n + 1) for _ in range(3)] for _ in range(n + 1)]
        
        # Initialize first round - Bob can play any creature
        for bob_idx in range(3):
            score = get_score(s[0], creatures[bob_idx])
            dp[1][bob_idx][n + score] = 1  # offset by n
        
        # Fill DP table for remaining rounds
        for i in range(1, n):
            for last in range(3):
                for diff in range(2*n + 1):
                    if dp[i][last][diff] == 0:
                        continue
                    
                    for bob_idx in range(3):
                        # Bob can't play same creature as previous round
                        if bob_idx == last:
                            continue
                        
                        score = get_score(s[i], creatures[bob_idx])
                        new_diff = diff + score
                        
                        if 0 <= new_diff <= 2*n:
                            dp[i+1][bob_idx][new_diff] = (dp[i+1][bob_idx][new_diff] + dp[i][last][diff]) % MOD
        
        # Count winning sequences (Bob's score > Alice's score)
        result = 0
        for last in range(3):
            for diff in range(n + 1, 2*n + 1):  # diff > n means Bob wins
                result = (result + dp[n][last][diff]) % MOD
        
        return result