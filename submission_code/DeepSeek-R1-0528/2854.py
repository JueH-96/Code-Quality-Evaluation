class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        total_length = sum(len(word) for word in words)
        dp = [[-10**9] * 26 for _ in range(26)]
        first_char0 = words[0][0]
        last_char0 = words[0][-1]
        l0 = ord(first_char0) - ord('a')
        r0 = ord(last_char0) - ord('a')
        dp[l0][r0] = 0
        
        for i in range(1, n):
            word = words[i]
            first_char = word[0]
            last_char = word[-1]
            c = ord(first_char) - ord('a')
            d = ord(last_char) - ord('a')
            new_dp = [[-10**9] * 26 for _ in range(26)]
            for l in range(26):
                for r in range(26):
                    if dp[l][r] == -10**9:
                        continue
                    savings_append = dp[l][r] + (1 if r == c else 0)
                    if new_dp[l][d] < savings_append:
                        new_dp[l][d] = savings_append
                    savings_prepend = dp[l][r] + (1 if d == l else 0)
                    if new_dp[c][r] < savings_prepend:
                        new_dp[c][r] = savings_prepend
            dp = new_dp
        
        max_savings = max(max(row) for row in dp)
        return total_length - max_savings