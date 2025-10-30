import math

class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        
        # Initialize DP array
        prev_dp = [math.inf] * 26
        original_char = ord(word[0]) - ord('a')
        for c in range(26):
            if c == original_char:
                prev_dp[c] = 0
            else:
                prev_dp[c] = 1
        
        for i in range(1, n):
            current_dp = [math.inf] * 26
            original = ord(word[i]) - ord('a')
            for c in range(26):
                current_cost = 0 if c == original else 1
                forbidden = set()
                if c > 0:
                    forbidden.add(c - 1)
                forbidden.add(c)
                if c < 25:
                    forbidden.add(c + 1)
                # Find the minimal previous state that is allowed
                min_prev = math.inf
                for prev_c in range(26):
                    if prev_c not in forbidden:
                        if prev_dp[prev_c] < min_prev:
                            min_prev = prev_dp[prev_c]
                if min_prev != math.inf:
                    current_dp[c] = min_prev + current_cost
            prev_dp = current_dp[:]
        
        return min(prev_dp)