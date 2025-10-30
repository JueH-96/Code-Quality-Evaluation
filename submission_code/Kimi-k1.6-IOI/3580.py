import numpy as np

class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        if m > n:
            return -1
        
        chars = 'abcdefghijklmnopqrstuvwxyz'
        cumulative = None
        
        for c in chars:
            s_c = np.array([1 if ch == c else 0 for ch in s], dtype=np.float64)
            p_c = np.array([1 if ch == c else 0 for ch in pattern], dtype=np.float64)
            reversed_p_c = p_c[::-1]
            
            # Compute convolution using FFT
            conv = np.fft.fftconvolve(s_c, reversed_p_c, mode='full')
            conv = np.round(np.real(conv)).astype(int)
            
            if cumulative is None:
                cumulative = conv
            else:
                cumulative += conv
        
        min_index = -1
        target_length = m
        for i in range(n - m + 1):
            matches = cumulative[i + m - 1]
            mismatches = target_length - matches
            if mismatches <= 1:
                min_index = i
                break
        
        return min_index