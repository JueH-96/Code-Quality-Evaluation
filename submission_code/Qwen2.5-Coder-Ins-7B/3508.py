class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n == k:
            return 0
        
        if n < k:
            return -1
        
        count = 0
        while n > k:
            if n & 1 != k & 1:
                count += 1
            n >>= 1
            k >>= 1
        
        return count + (n - k)