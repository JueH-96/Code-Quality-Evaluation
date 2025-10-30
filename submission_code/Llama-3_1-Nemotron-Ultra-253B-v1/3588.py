MOD = 10**9 + 7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        # Convert the string s into a list of indices: F->0, W->1, E->2
        s_list = [0 if c == 'F' else 1 if c == 'W' else 2 for c in s]
        # Precompute the delta table
        delta_table = [
            [0, 1, -1],  # a=0 (F)
            [-1, 0, 1],  # a=1 (W)
            [1, -1, 0]   # a=2 (E)
        ]
        # Initialize DP: dp[i] is a dictionary of (prev, delta) -> count
        dp = [{} for _ in range(n + 1)]
        # Initial state: prev=3 (None), delta=0, count=1
        dp[0][(3, 0)] = 1
        for i in range(n):
            current_dp = dp[i]
            next_dp = {}
            a = s_list[i]
            for (prev, delta), cnt in current_dp.items():
                for curr in [0, 1, 2]:
                    # Check if current curr is allowed (not same as prev unless prev is None)
                    if prev != 3 and curr == prev:
                        continue
                    # Calculate the delta contribution for this round
                    delta_contribution = delta_table[a][curr]
                    new_delta = delta + delta_contribution
                    new_prev = curr
                    key = (new_prev, new_delta)
                    # Update the next_dp
                    if key in next_dp:
                        next_dp[key] = (next_dp[key] + cnt) % MOD
                    else:
                        next_dp[key] = cnt % MOD
            dp[i + 1] = next_dp
        # Sum all counts where delta > 0
        total = 0
        for (prev, delta), cnt in dp[n].items():
            if delta > 0:
                total = (total + cnt) % MOD
        return total