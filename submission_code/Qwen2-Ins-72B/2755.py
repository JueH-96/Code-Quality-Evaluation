class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        d = set(dictionary)
        n = len(s)
        @lru_cache(None)
        def dp(i):
            if i >= n: return 0
            ans = dp(i+1) + 1
            for j in range(i, n):
                if s[i:j+1] in d:
                    ans = min(ans, dp(j+1))
            return ans
        return dp(0)