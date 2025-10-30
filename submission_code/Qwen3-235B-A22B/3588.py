class Solution:
    def countWinningSequences(self, s: str) -> int:
        mod = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute delta_table [a][b] where a is Alice's move, b is Bob's move
        delta_table = [
            # F, W, E
            [0, 1, -1],  # Alice's F
            [-1, 0, 1],  # Alice's W
            [1, -1, 0],  # Alice's E
        ]
        
        # Initialize current_dp for the first round
        current_dp = [[0] * 2001 for _ in range(3)]
        a_char = s[0]
        a_idx = {'F': 0, 'W': 1, 'E': 2}[a_char]
        for b_idx in range(3):
            delta = delta_table[a_idx][b_idx]
            diff_idx = delta + 1000
            if 0 <= diff_idx <= 2000:
                current_dp[b_idx][diff_idx] = 1
        
        # Iterate through the remaining rounds
        for i in range(1, n):
            a_char = s[i]
            a_idx = {'F': 0, 'W': 1, 'E': 2}[a_char]
            next_dp = [[0] * 2001 for _ in range(3)]
            
            for prev in range(3):  # previous move
                for diff_idx in range(2001):
                    count = current_dp[prev][diff_idx]
                    if count == 0:
                        continue
                    current_diff = diff_idx - 1000
                    
                    # Try all possible current moves different from prev
                    for curr in range(3):
                        if curr == prev:
                            continue
                        # Calculate delta for this round
                        delta = delta_table[a_idx][curr]
                        new_diff = current_diff + delta
                        new_diff_idx = new_diff + 1000
                        if new_diff_idx < 0 or new_diff_idx > 2000:
                            continue
                        next_dp[curr][new_diff_idx] = (next_dp[curr][new_diff_idx] + count) % mod
            
            current_dp = next_dp
        
        # Sum all states where diff > 0
        result = 0
        for prev in range(3):
            for diff_idx in range(1001, 2001):  # diff >=1
                result = (result + current_dp[prev][diff_idx]) % mod
        
        return result