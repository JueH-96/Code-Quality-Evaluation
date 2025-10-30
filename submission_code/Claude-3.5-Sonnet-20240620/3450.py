class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        cycle = 2 * (n - 1)
        k %= cycle
        
        if k < n:
            return k
        else:
            return 2 * (n - 1) - k