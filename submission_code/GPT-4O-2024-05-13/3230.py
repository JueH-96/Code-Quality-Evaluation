class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def is_almost_equal(a, b):
            return a == b or abs(ord(a) - ord(b)) == 1
        
        n = len(word)
        if n <= 1:
            return 0
        
        dp = [0] * n
        
        for i in range(1, n):
            dp[i] = dp[i-1]
            if is_almost_equal(word[i], word[i-1]):
                dp[i] += 1
                if i > 1:
                    dp[i] = min(dp[i], dp[i-2] + 1)
        
        return dp[-1]

# Example usage:
# sol = Solution()
# print(sol.removeAlmostEqualCharacters("aaaaa"))  # Output: 2
# print(sol.removeAlmostEqualCharacters("abddez"))  # Output: 2
# print(sol.removeAlmostEqualCharacters("zyxyxyz"))  # Output: 3