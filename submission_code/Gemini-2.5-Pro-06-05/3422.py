class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the array with n ones, representing the state at t=0.
        a = [1] * n
        
        # Simulate the process for k seconds.
        for _ in range(k):
            # Each second, the array is updated to its prefix sums.
            # The update rule for a_new[i] is sum(a_old[0]...a_old[i]).
            # This can be simplified to a_new[i] = a_new[i-1] + a_old[i].
            # This calculation can be performed in-place efficiently.
            for i in range(1, n):
                # a[i-1] already holds the new value for the current second.
                a[i] = (a[i] + a[i-1]) % MOD
                
        # The required value is the last element of the array after k seconds.
        return a[n - 1]