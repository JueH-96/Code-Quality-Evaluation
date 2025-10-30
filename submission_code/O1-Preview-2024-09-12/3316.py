class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10 ** 9 + 7
        n = len(nums)
        nums.sort()
        differences = set()
        differences.add(0)  # Include zero difference as in test case 2
        for i in range(n):
            for j in range(i+1, n):
                differences.add(nums[j] - nums[i])
        differences = sorted(differences)
        S = len(differences)
        # Precompute next_valid[D][i]
        next_valid = {}
        for D in differences:
            next_valid[D] = [0]*n
            for i in range(n):
                # Find the minimal index j > i such that nums[j] - nums[i] >= D
                j = i + 1
                while j < n and nums[j] - nums[i] < D:
                    j += 1
                next_valid[D][i] = j
        f = {}
        # Initialize dp
        dp = [ [0]*(k+1) for _ in range(n+1) ]
        total = 0
        f_next = 0
        for D in sorted(differences, reverse=True):
            # Reset dp for current D
            for i in range(n+1):
                for l in range(k+1):
                    dp[i][l] = 0
            for i in range(n+1):
                dp[i][0] = 1  # Base case: choosing 0 elements
            for i in range(n-1, -1, -1):
                for l in range(1, k+1):
                    # Skip nums[i]
                    dp[i][l] = dp[i+1][l] % MOD
                    # Pick nums[i]
                    j = next_valid[D][i]
                    if j <= n:
                        dp[i][l] = (dp[i][l] + dp[j][l-1]) % MOD
            f_current = dp[0][k] % MOD
            total = (total + (f_current - f_next) * D) % MOD
            f_next = f_current
        return total % MOD