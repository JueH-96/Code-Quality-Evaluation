class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        # Initialize array a with all 1s
        a = [1] * n
        
        # Perform k updates
        for _ in range(k):
            # Compute prefix sums in-place
            for i in range(1, n):
                a[i] = (a[i] + a[i-1]) % mod
        
        # The answer is the last element after k seconds
        return a[n-1]