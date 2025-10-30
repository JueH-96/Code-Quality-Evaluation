class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def check(start):
            i = 0
            for j in range(start, start + len(pattern)):
                if s[j] != pattern[i]:
                    if i != j: return False
                    i += 1
                i += 1
            return True
        
        l, r = 0, len(pattern)
        while l < r:
            m = (l + r) // 2
            if check(m):
                r = m
            else:
                l = m + 1
        return l if l < len(s) else -1