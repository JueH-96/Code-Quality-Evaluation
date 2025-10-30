class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        diction = set(dictionary)
        @cache
        def dp(i):
            if not i:
                return 0
            if s[:i] in diction:
                return dp(i-1)
            return min([dp(c_end) + 1 for c_end in range(i)] + [dp(i - len(w)) for w in dictionary if s.endswith(w, 0, i)])
        return dp(len(s))