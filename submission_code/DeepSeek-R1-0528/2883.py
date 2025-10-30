class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        max_limit = (1 << 15) - 1
        power_set = set()
        power = 1
        while power <= max_limit:
            bin_rep = bin(power)[2:]
            power_set.add(bin_rep)
            power *= 5
        
        max_len = max(len(bin_rep) for bin_rep in power_set) if power_set else 0
        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            if s[i] == '0':
                continue
            current = ""
            for j in range(i, n):
                current += s[j]
                if len(current) > max_len:
                    break
                if current in power_set:
                    dp[j + 1] = min(dp[j + 1], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1