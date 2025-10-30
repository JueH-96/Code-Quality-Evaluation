class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        dict_set = set(dictionary)
        n = len(s)
        memo = {}
        def dp(i):
            if i == n:
                return 0
            if i in memo:
                return memo[i]
            # Assume s[i] is extra character
            res = dp(i+1) + 1
            # Try all substrings starting at i
            for j in range(i, n):
                substr = s[i:j+1]
                if substr in dict_set:
                    res = min(res, dp(j+1))
            memo[i] = res
            return res
        return dp(0)