class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        # dp[i] will store the minimum number of substrings needed for s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # Helper function to check if binary string "sub" is a power of 5.
        def is_power_of_five(sub: str) -> bool:
            # Cannot have leading zeros unless it's a single "0" (which is not valid anyway because 0 is not power of 5)
            if sub[0] == '0':
                return False
            num = int(sub, 2)
            # Check if num is a power of 5.
            # 5^0 = 1 is valid.
            while num % 5 == 0:
                num //= 5
                if num == 1:
                    return True
            return False

        # Check every possible partition
        for i in range(1, n + 1):
            for j in range(0, i):
                substr = s[j:i]
                if is_power_of_five(substr):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1

# For quick local testing:
if __name__ == "__main__":
    sol = Solution()
    # Provided examples:
    print(sol.minimumBeautifulSubstrings("1011"))  # Expected Output: 2 -> partition is ["101", "1"]
    print(sol.minimumBeautifulSubstrings("111"))   # Expected Output: 3 -> partition is ["1", "1", "1"]
    print(sol.minimumBeautifulSubstrings("0"))     # Expected Output: -1