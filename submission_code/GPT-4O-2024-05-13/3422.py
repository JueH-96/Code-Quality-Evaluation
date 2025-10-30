class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the array with 1s
        a = [1] * n
        
        # Update the array k times
        for _ in range(k):
            for i in range(1, n):
                a[i] = (a[i] + a[i-1]) % MOD
        
        return a[-1]