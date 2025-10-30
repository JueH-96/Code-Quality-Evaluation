class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Function to check if a binary string is a power of 5
        def is_power_of_five(binary_str):
            num = int(binary_str, 2)
            while num > 1:
                if num % 5 != 0:
                    return False
                num //= 5
            return num == 1

        # Dynamic Programming approach
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                if s[j] != '0' and is_power_of_five(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1

# Example usage:
# sol = Solution()
# print(sol.minimumBeautifulSubstrings("1011"))  # Output: 2
# print(sol.minimumBeautifulSubstrings("111"))   # Output: 3
# print(sol.minimumBeautifulSubstrings("0"))     # Output: -1