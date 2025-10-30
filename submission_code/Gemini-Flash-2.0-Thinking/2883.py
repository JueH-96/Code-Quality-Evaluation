class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        beautiful_binaries = set()
        powers = [1]
        while True:
            binary_repr = bin(powers[-1])[2:]
            if len(binary_repr) > n:
                break
            beautiful_binaries.add(binary_repr)
            powers.append(powers[-1] * 5)

        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if not substring.startswith('0') and substring in beautiful_binaries:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1