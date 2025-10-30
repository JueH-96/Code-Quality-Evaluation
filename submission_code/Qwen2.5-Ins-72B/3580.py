class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def is_almost_equal(sub: str, pattern: str) -> bool:
            diff_count = sum(1 for x, y in zip(sub, pattern) if x != y)
            return diff_count <= 1
        
        n, m = len(s), len(pattern)
        
        for i in range(n - m + 1):
            if is_almost_equal(s[i:i + m], pattern):
                return i
        
        return -1