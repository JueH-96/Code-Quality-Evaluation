class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        # Helper: minimal cost to change character a into character b using allowed operations.
        # The cost is the minimal number of steps in the circular alphabet.
        def change_cost(a: str, b: str) -> int:
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff)
        
        from functools import lru_cache
        
        n = len(s)
        
        # dp(i, j, rem) returns the maximum length of a palindromic subsequence that can be 
        # achieved from the substring s[i..j] using at most rem operations.
        @lru_cache(maxsize=None)
        def dp(i, j, rem):
            if i > j:
                return 0
            if i == j:
                # A single letter is a palindrome of length 1.
                return 1
            best = 0
            # Option: skip i
            best = max(best, dp(i+1, j, rem))
            # Option: skip j
            best = max(best, dp(i, j-1, rem))
            # Option: use both characters at positions i and j
            cost = change_cost(s[i], s[j])
            if cost <= rem:
                best = max(best, 2 + dp(i+1, j-1, rem - cost))
            return best
        
        return dp(0, n-1, k)
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.longestPalindromicSubsequence("abced", 2))  # Expected output: 3
    # Example 2:
    print(sol.longestPalindromicSubsequence("aaazzz", 4))  # Expected output: 6