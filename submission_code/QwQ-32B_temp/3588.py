class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        
        chars = ['F', 'W', 'E']
        allowed = [[1, 2], [0, 2], [0, 1]]  # allowed next moves for each previous character
        
        # Initialize DP arrays
        prev_dp = [[0] * 2001 for _ in range(3)]
        
        # First character processing
        first_char = s[0]
        for move in range(3):
            b_char = chars[move]
            a_char = first_char
            if a_char == b_char:
                delta = 0
            else:
                if (a_char == 'F' and b_char == 'E') or (a_char == 'W' and b_char == 'F') or (a_char == 'E' and b_char == 'W'):
                    delta = -1
                else:
                    delta = 1
            shifted_diff = delta + 1000
            prev_dp[move][shifted_diff] += 1
        
        # Process remaining rounds
        for i in range(1, n):
            current_dp = [[0] * 2001 for _ in range(3)]
            current_alice_char = s[i]
            for prev_char in range(3):
                for shifted_diff in range(2001):
                    count = prev_dp[prev_char][shifted_diff]
                    if count == 0:
                        continue
                    prev_diff = shifted_diff - 1000
                    for next_char in allowed[prev_char]:
                        b_char = chars[next_char]
                        a_char = current_alice_char
                        if a_char == b_char:
                            delta_i = 0
                        else:
                            if (a_char == 'F' and b_char == 'E') or (a_char == 'W' and b_char == 'F') or (a_char == 'E' and b_char == 'W'):
                                delta_i = -1
                            else:
                                delta_i = 1
                        new_diff = prev_diff + delta_i
                        shifted_new = new_diff + 1000
                        current_dp[next_char][shifted_new] = (current_dp[next_char][shifted_new] + count) % MOD
            prev_dp = current_dp
        
        # Calculate the total valid sequences
        total = 0
        for prev_char in range(3):
            for shifted_diff in range(2001):
                if (shifted_diff - 1000) > 0:
                    total = (total + prev_dp[prev_char][shifted_diff]) % MOD
        return total