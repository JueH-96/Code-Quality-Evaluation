import math
from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        counts = Counter(s)
        freqs = sorted(counts.values(), reverse=True)
        m = len(freqs)
        min_ops = float('inf')
        
        for t in range(1, m + 1):
            current_freqs = freqs[:t]
            sum_t = sum(current_freqs)
            sum_deleted = sum(freqs[t:]) if t < m else 0
            avg = sum_t / t
            k_floor = math.floor(avg)
            k_ceil = math.ceil(avg)
            
            candidates = {k_floor, k_ceil}
            current_min = float('inf')
            
            for k in candidates:
                A = 0
                B = 0
                for f in current_freqs:
                    if f > k:
                        A += (f - k)
                    elif f < k:
                        B += (k - f)
                current_min = min(current_min, max(A, B))
            
            total_cost = sum_deleted + current_min
            if total_cost < min_ops:
                min_ops = total_cost
        
        return min_ops