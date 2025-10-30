class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        char_to_idx = {'F': 0, 'W': 1, 'E': 2}
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute the delta table: delta[Alice's move][Bob's move] = point difference (Bob - Alice)
        delta = [
            [0, 1, -1],    # F vs F, W, E
            [1, 0, -1],    # W vs F, W, E
            [-1, 1, 0]     # E vs F, W, E
        ]
        
        # Initialize previous DP for the first round
        prev_dp = [[0] * 2001 for _ in range(3)]
        a_idx = char_to_idx[s[0]]
        for b_idx in range(3):
            d = delta[a_idx][b_idx]
            diff_index = d + 1000
            prev_dp[b_idx][diff_index] = 1
        
        # Iterate through the remaining rounds
        for i in range(1, n):
            curr_dp = [[0] * 2001 for _ in range(3)]
            a_idx = char_to_idx[s[i]]
            for prev_move in range(3):
                for diff_index in range(2001):
                    count = prev_dp[prev_move][diff_index]
                    if count == 0:
                        continue
                    current_diff = diff_index - 1000
                    for next_move in range(3):
                        if next_move == prev_move:
                            continue
                        new_d = delta[a_idx][next_move]
                        new_diff = current_diff + new_d
                        new_diff_index = new_diff + 1000
                        curr_dp[next_move][new_diff_index] = (curr_dp[next_move][new_diff_index] + count) % MOD
            prev_dp = curr_dp
        
        # Sum all cases where the difference is positive
        total = 0
        for b_idx in range(3):
            for diff_index in range(2001):
                if diff_index > 1000:
                    total = (total + prev_dp[b_idx][diff_index]) % MOD
        return total