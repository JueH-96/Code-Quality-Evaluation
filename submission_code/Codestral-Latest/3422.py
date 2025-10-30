class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        # Initialize the array with all ones
        a = [1] * n

        # Function to update the array
        def update_array(arr):
            new_arr = [0] * n
            for i in range(n):
                new_arr[i] = sum(arr[:i+1]) % MOD
            return new_arr

        # Update the array k times
        for _ in range(k):
            a = update_array(a)

        return a[-1]