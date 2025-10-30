class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5_bin = []
        power = 0
        while True:
            val = 5**power
            binary_val = bin(val)[2:]
            if len(binary_val) > len(s):
                break
            powers_of_5_bin.append(binary_val)
            power += 1
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0' and dp[i] == float('inf'):
                continue
            for b in powers_of_5_bin:
                len_b = len(b)
                if i + len_b <= n:
                    substring = s[i:i+len_b]
                    if substring == b:
                        dp[i] = min(dp[i], 1 + dp[i + len_b])
                        
        if dp[0] == float('inf'):
            return -1
        else:
            return dp[0]