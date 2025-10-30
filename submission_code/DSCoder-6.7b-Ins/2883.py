class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        powers_of_5 = [1]
        for _ in range(n):
            powers_of_5.append(powers_of_5[-1] * 5)
        binary_representations = set(powers_of_5)

        for i in range(1, n + 1):
            for j in range(i):
                if s[j:i] not in binary_representations:
                    continue
                dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1] if dp[-1] < float('inf') else -1