class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the array with 1s
        a = [1] * n
        
        # Update the array for k seconds
        for _ in range(k):
            for j in range(1, n):
                a[j] = (a[j] + a[j - 1]) % MOD
        
        return a[n - 1]