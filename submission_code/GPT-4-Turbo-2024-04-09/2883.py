class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Helper function to check if a binary string is a power of 5
        def is_power_of_5(binary_str):
            if binary_str[0] == '0':
                return False
            num = int(binary_str, 2)
            while num % 5 == 0:
                num //= 5
            return num == 1
        
        # Dynamic programming array to store the minimum number of partitions
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Fill the dp array
        for i in range(1, n + 1):
            for j in range(i):
                if is_power_of_5(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # If dp[n] is still inf, it means it's impossible to partition
        return dp[n] if dp[n] != float('inf') else -1