class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        
        chars = [ord(c) - ord('a') for c in word]
        prev_dp = [float('inf')] * 26
        
        for c in range(26):
            if c == chars[0]:
                prev_dp[c] = 0
            else:
                prev_dp[c] = 1
        
        for i in range(1, n):
            current_dp = [float('inf')] * 26
            current_char = chars[i]
            for c in range(26):
                min_cost = float('inf')
                for p in range(26):
                    if abs(p - c) >= 2:
                        cost = prev_dp[p] + (0 if c == current_char else 1)
                        if cost < min_cost:
                            min_cost = cost
                current_dp[c] = min_cost
            prev_dp = current_dp
        
        return min(prev_dp)