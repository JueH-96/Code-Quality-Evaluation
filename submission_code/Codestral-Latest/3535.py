class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Initialize dp arrays for increasing and decreasing sequences
        inc = [[0] * (nums[i] + 1) for i in range(n)]
        dec = [[0] * (nums[i] + 1) for i in range(n)]

        # Base case: for the first element
        for j in range(nums[0] + 1):
            inc[0][j] = 1
            dec[0][j] = 1

        # Fill the dp arrays
        for i in range(1, n):
            for j in range(nums[i] + 1):
                # Calculate increasing sequences
                for k in range(j + 1):
                    inc[i][j] = (inc[i][j] + inc[i - 1][k]) % MOD
                # Calculate decreasing sequences
                for k in range(j, nums[i] + 1):
                    dec[i][j] = (dec[i][j] + dec[i - 1][k]) % MOD

        # Calculate the total number of monotonic pairs
        total_pairs = 0
        for i in range(n):
            for j in range(nums[i] + 1):
                total_pairs = (total_pairs + (inc[i][j] * dec[i][nums[i] - j]) % MOD) % MOD

        return total_pairs