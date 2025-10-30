class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        # Increase recursion limit in case it's needed for deep recursions.
        import sys
        sys.setrecursionlimit(10000)
        from functools import lru_cache

        n = len(s)
        
        # Helper function to compute the minimal cost to change two characters 
        # into the same letter.
        # This cost is exactly the circular distance between the two letters.
        def pairCost(a, b):
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff)
        
        @lru_cache(maxsize=None)
        def dp(l, r, rem):
            if l > r:
                return 0
            if l == r:
                # Only one character: it can serve as the center of a palindrome.
                return 1
            
            # Option 1: skip the left character.
            best = dp(l + 1, r, rem)
            # Option 2: skip the right character.
            best = max(best, dp(l, r - 1, rem))
            
            # Option 3: try to pair s[l] with s[r] if we have enough operations.
            cost = pairCost(s[l], s[r])
            if rem >= cost:
                best = max(best, 2 + dp(l + 1, r - 1, rem - cost))
            return best
        
        return dp(0, n - 1, k)