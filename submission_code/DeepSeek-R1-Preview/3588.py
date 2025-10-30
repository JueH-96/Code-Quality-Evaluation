class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        a = []
        for c in s:
            if c == 'F':
                a.append(0)
            elif c == 'W':
                a.append(1)
            else:
                a.append(2)
        
        delta_matrix = [
            [0, 1, -1],   # a is F (0)
            [-1, 0, 1],   # a is W (1)
            [1, -1, 0]    # a is E (2)
        ]
        
        shift = n  # Maximum difference is n
        max_diff = n
        prev_dp = [[0] * (2 * shift + 1) for _ in range(3)]
        
        # Initialize for the first round
        for b in range(3):
            delta = delta_matrix[a[0]][b]
            d_shifted = delta + shift
            if 0 <= d_shifted < 2 * shift + 1:
                prev_dp[b][d_shifted] = 1
        
        for i in range(1, n):
            current_dp = [[0] * (2 * shift + 1) for _ in range(3)]
            for prev_b in range(3):
                for d_shifted in range(2 * shift + 1):
                    count = prev_dp[prev_b][d_shifted]
                    if count == 0:
                        continue
                    prev_d = d_shifted - shift
                    for new_b in range(3):
                        if new_b == prev_b:
                            continue
                        current_a = a[i]
                        delta = delta_matrix[current_a][new_b]
                        new_d = prev_d + delta
                        new_d_shifted = new_d + shift
                        if 0 <= new_d_shifted < 2 * shift + 1:
                            current_dp[new_b][new_d_shifted] = (current_dp[new_b][new_d_shifted] + count) % MOD
            prev_dp = current_dp
        
        result = 0
        for b in range(3):
            for d_shifted in range(shift + 1, 2 * shift + 1):
                result = (result + prev_dp[b][d_shifted]) % MOD
        return result