class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        if n == 2:
            return k % 2
        
        cycle_len = (n - 1) * 2
        k %= cycle_len
        
        if k <= n - 1:
            return k
        else:
            return n - 1 - (k - (n - 1))