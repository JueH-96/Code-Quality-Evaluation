class Solution:
    def minOperations(self, k: int) -> int:
        if k==1: return 0
        m = k // 2
        # brute force simulation
        ops = 0 
        n = 1 
        v = 1 
        while n * v < k-1: 
            v += 1 
            n *= 2
            ops += 2
        # use binary search + linear search over the remaining interval to find the last operation 
        lo, hi = 1, m 
        while lo < hi: 
            mid = (lo + hi) // 2 
            if 2 * (v + mid) * (n // 2) >= k-1:  
                hi = mid 
            else: 
                lo = mid + 1 
        return ops + lo if 2 * (v + lo) * (n // 2) >= k-1 else ops + lo + 1