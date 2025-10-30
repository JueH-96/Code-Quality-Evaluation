class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the array with 1s
        a = [1] * n
        
        # Perform the operation for k seconds
        for _ in range(k):
            # Update the array
            for i in range(1, n):
                a[i] = (a[i] + a[i - 1]) % MOD
        
        # Return the last element of the array
        return a[n - 1]