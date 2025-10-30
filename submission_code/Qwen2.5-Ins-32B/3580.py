class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def is_almost_equal(sub, pattern):
            diff = 0
            for i in range(len(pattern)):
                if sub[i] != pattern[i]:
                    diff += 1
                    if diff > 1:
                        return False
            return True
        
        for i in range(len(s) - len(pattern) + 1):
            if is_almost_equal(s[i:i+len(pattern)], pattern):
                return i
        return -1