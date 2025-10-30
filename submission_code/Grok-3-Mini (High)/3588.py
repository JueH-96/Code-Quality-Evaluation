class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 1000000007
        n = len(s)
        OFFSET = n
        # dp[i][diff_idx][last_idx]: number of ways before playing round i
        dp = [[[0 for _ in range(4)] for _ in range(2 * n + 1)] for _ in range(n + 1)]
        dp[0][OFFSET][3] = 1  # Start with diff 0, no move
        
        # Delta matrix: delta[alice_idx][bob_move]
        delta = [
            [0, 1, -1],    # Alice 'F'
            [-1, 0, 1],    # Alice 'W'
            [1, -1, 0]     # Alice 'E'
        ]
        
        for i in range(n):  # For each round
            alice_char = s[i]
            alice_idx = 0 if alice_char == 'F' else 1 if alice_char == 'W' else 2  # Map to index
            
            for curr_diff_idx in range(2 * n + 1):
                for curr_last_idx in range(4):
                    val = dp[i][curr_diff_idx][curr_last_idx]
                    if val == 0:
                        continue  # Skip if no ways to this state
                    for bob_move in range(3):  # Bob's move: 0='F', 1='W', 2='E'
                        if curr_last_idx == 3 or bob_move != curr_last_idx:  # No consecutive same move
                            delta_diff = delta[alice_idx][bob_move]
                            new_diff_idx = curr_diff_idx + delta_diff
                            if 0 <= new_diff_idx <= 2 * n:
                                new_last_idx = bob_move
                                dp[i + 1][new_diff_idx][new_last_idx] = (dp[i + 1][new_diff_idx][new_last_idx] + val) % MOD
        
        # Sum over all states where diff > 0, i.e., diff_idx > OFFSET
        ans = 0
        for diff_idx in range(OFFSET + 1, 2 * n + 1):  # diff_idx from n+1 to 2n
            for last_idx in range(3):  # last move: 0,1,2
                ans = (ans + dp[n][diff_idx][last_idx]) % MOD
        
        return ans