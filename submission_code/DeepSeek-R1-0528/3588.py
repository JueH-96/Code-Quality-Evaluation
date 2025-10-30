class Solution:
    def countWinningSequences(self, s: str) -> int:
        mod = 10**9 + 7
        n_rounds = len(s)
        if n_rounds == 0:
            return 0
        char_to_index = {'F': 0, 'W': 1, 'E': 2}
        gain_matrix = [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ]
        offset = n_rounds
        total_size = 2 * n_rounds + 1
        
        dp_prev = [[0] * total_size for _ in range(3)]
        first_char = s[0]
        idx0 = char_to_index[first_char]
        for j in range(3):
            g = gain_matrix[j][idx0]
            d_index = g + offset
            if 0 <= d_index < total_size:
                dp_prev[j][d_index] = (dp_prev[j][d_index] + 1) % mod
        
        for i in range(1, n_rounds):
            dp_curr = [[0] * total_size for _ in range(3)]
            current_char = s[i]
            idx_c = char_to_index[current_char]
            gains_this_round = [gain_matrix[j][idx_c] for j in range(3)]
            for j in range(3):
                g = gains_this_round[j]
                for k in range(3):
                    if j == k:
                        continue
                    for d_index_prev in range(total_size):
                        count = dp_prev[k][d_index_prev]
                        if count:
                            new_index = d_index_prev + g
                            if 0 <= new_index < total_size:
                                dp_curr[j][new_index] = (dp_curr[j][new_index] + count) % mod
            dp_prev = dp_curr
        
        ans = 0
        for j in range(3):
            for idx in range(offset + 1, total_size):
                ans = (ans + dp_prev[j][idx]) % mod
        return ans