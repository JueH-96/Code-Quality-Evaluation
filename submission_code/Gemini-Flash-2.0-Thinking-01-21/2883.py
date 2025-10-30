class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5_bin = set()
        power = 1
        while True:
            bin_str = bin(power)[2:]
            if len(bin_str) > 15:
                break
            powers_of_5_bin.add(bin_str)
            power *= 5

        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                sub = s[j:i]
                if len(sub) > 1 and sub[0] == '0':
                    continue
                if sub == '0':
                    continue
                if sub in powers_of_5_bin:
                    if dp[j] != float('inf'):
                        dp[i] = min(dp[i], dp[j] + 1)

        if dp[n] == float('inf'):
            return -1
        return dp[n]