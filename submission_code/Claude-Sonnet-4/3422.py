class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize array with all 1s
        a = [1] * n
        
        # Simulate k seconds
        for _ in range(k):
            # Calculate prefix sums
            for i in range(1, n):
                a[i] = (a[i] + a[i-1]) % MOD
        
        return a[n-1]