class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def is_almost_equal(substr, pattern):
            if len(substr) != len(pattern):
                return False
            diff = 0
            for a, b in zip(substr, pattern):
                if a != b:
                    diff += 1
                    if diff > 1:
                        return False
            return True

        n, m = len(s), len(pattern)
        
        if m > n:
            return -1
        
        for i in range(n - m + 1):
            if is_almost_equal(s[i:i+m], pattern):
                return i
        
        return -1