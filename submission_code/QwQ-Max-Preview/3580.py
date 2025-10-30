import numpy as np

class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        if m > n:
            return -1
        
        unique_chars = set(pattern)
        total = np.zeros(n - m + 1, dtype=np.int32)
        
        for c in unique_chars:
            s_mask = np.array([1 if ch == c else 0 for ch in s], dtype=np.int32)
            p_mask = np.array([1 if ch == c else 0 for ch in pattern], dtype=np.int32)
            corr = np.correlate(s_mask, p_mask, mode='valid')
            total += corr
        
        for i in range(len(total)):
            if (m - total[i]) <= 1:
                return i
        
        return -1