class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        i = 0
        j = n - 1
        while i < j:
            while i < j and s[i] == s[i + 1]:
                i += 1
            while i < j and s[j] == s[j - 1]:
                j -= 1
            if i < j:
                cost += min(i + 1, n - j)
                if s[i] == s[j]:
                    i += 1
                else:
                    j -= 1
        return cost