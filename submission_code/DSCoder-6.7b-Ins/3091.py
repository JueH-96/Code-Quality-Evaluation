class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        prefix = [0] * (n + 1)
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = (prefix[i - 1] + nums[i]) % MOD
        for i in range(1, n + 1):
            for j in range(i):
                if nums[j] > nums[i - 1]:
                    dp[i] = (dp[i] + dp[j]) % MOD
        count = [0] * (n + 1)
        count[0] = 1
        for i in range(1, n + 1):
            for j in range(i):
                if nums[j] <= nums[i - 1]:
                    count[i] = (count[i] + count[j]) % MOD
        sum_count = [0] * (n + 1)
        sum_count[0] = count[0]
        for i in range(1, n + 1):
            sum_count[i] = (sum_count[i - 1] + count[i]) % MOD
        l -= 1
        r -= 1
        while l >= 0 and r >= 0:
            l -= 1
            r -= 1
            res = (sum_count[r] - (sum_count[l] if l >= 0 else 0) + MOD) % MOD
            return res