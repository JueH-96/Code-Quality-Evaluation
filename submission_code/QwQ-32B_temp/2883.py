class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Generate all binary representations of powers of 5 up to 5^6 (since 5^7 exceeds 2^15)
        power_set = set()
        current = 1  # 5^0
        while current <= 2 ** 15:
            power_set.add(bin(current)[2:])  # Convert to binary without '0b' prefix
            current *= 5
        
        n = len(s)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                substr = s[j:i]
                if substr in power_set:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n] if dp[n] != INF else -1