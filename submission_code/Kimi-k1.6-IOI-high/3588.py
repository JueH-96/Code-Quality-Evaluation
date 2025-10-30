class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        
        char_to_idx = {'F': 0, 'W': 1, 'E': 2}
        wins = {'F': 'E', 'W': 'F', 'E': 'W'}
        current_dp = [[0] * 2001 for _ in range(3)]
        
        # Initialize for the first character
        first_char = s[0]
        for b_idx in range(3):
            b_char = ['F', 'W', 'E'][b_idx]
            if wins[first_char] == b_char:
                delta = -1
            elif wins[b_char] == first_char:
                delta = 1
            else:
                delta = 0
            diff = delta
            diff_index = diff + 1000
            current_dp[b_idx][diff_index] = (current_dp[b_idx][diff_index] + 1) % MOD
        
        # Process remaining characters
        for i in range(1, n):
            a_char = s[i]
            next_dp = [[0] * 2001 for _ in range(3)]
            for last_char_idx in range(3):
                for diff in range(-1000, 1001):
                    count = current_dp[last_char_idx][diff + 1000]
                    if count == 0:
                        continue
                    for next_char_idx in range(3):
                        if next_char_idx == last_char_idx:
                            continue
                        b_char = ['F', 'W', 'E'][next_char_idx]
                        if wins[a_char] == b_char:
                            delta = -1
                        elif wins[b_char] == a_char:
                            delta = 1
                        else:
                            delta = 0
                        new_diff = diff + delta
                        new_diff_index = new_diff + 1000
                        if 0 <= new_diff_index < 2001:
                            next_dp[next_char_idx][new_diff_index] = (next_dp[next_char_idx][new_diff_index] + count) % MOD
            current_dp = next_dp
        
        # Sum all valid sequences where diff > 0
        total = 0
        for last_char_idx in range(3):
            for diff in range(1, 1001):  # diff ranges from 1 to 1000 inclusive
                total = (total + current_dp[last_char_idx][diff + 1000]) % MOD
        return total