class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        INF = 10**9
        prev_dp = [INF] * 26
        orig0 = ord(word[0]) - ord('a')
        for c in range(26):
            prev_dp[c] = 0 if c == orig0 else 1
        
        for i in range(1, n):
            current_dp = [INF] * 26
            orig = ord(word[i]) - ord('a')
            for c in range(26):
                cost = 0 if c == orig else 1
                for pc in range(26):
                    if abs(c - pc) <= 1:
                        continue
                    if prev_dp[pc] + cost < current_dp[c]:
                        current_dp[c] = prev_dp[pc] + cost
            prev_dp = current_dp
        
        return min(prev_dp)