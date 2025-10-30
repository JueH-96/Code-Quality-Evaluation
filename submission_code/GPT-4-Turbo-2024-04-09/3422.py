class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Create the initial array of size n with all elements as 1
        a = [1] * n
        
        # Perform the update k times
        for _ in range(k):
            # Update the array from the end to the start
            for i in range(1, n):
                a[i] = (a[i] + a[i-1]) % MOD
        
        # Return the last element of the array after k seconds
        return a[-1]