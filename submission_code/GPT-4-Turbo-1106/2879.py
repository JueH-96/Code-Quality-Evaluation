class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        def cost_to_semi_palindrome(sub):
            n = len(sub)
            cost = 0
            for d in range(1, n):
                if n % d == 0:
                    for i in range(d):
                        count = {}
                        for j in range(i, n, d):
                            count[sub[j]] = count.get(sub[j], 0) + 1
                        max_freq = max(count.values())
                        cost += (n // d - max_freq)
                    return cost // (n // d)
            return cost

        n = len(s)
        if k == 1:
            return cost_to_semi_palindrome(s)

        # Initialize dp array where dp[i][j] represents the minimum cost to partition
        # the first i characters into j substrings.
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for x in range(i):
                    dp[i][j] = min(dp[i][j], dp[x][j - 1] + cost_to_semi_palindrome(s[x:i]))

        return dp[n][k]

# Example usage:
# sol = Solution()
# print(sol.minimumChanges("abcac", 2))  # Output: 1
# print(sol.minimumChanges("abcdef", 2))  # Output: 2
# print(sol.minimumChanges("aabbaa", 3))  # Output: 0