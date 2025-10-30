from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        max_d = nums[-1] - nums[0]
        critical_ds = set()
        for i in range(n):
            for j in range(i+1, n):
                critical_ds.add(nums[j] - nums[i])
        critical_ds.add(0)
        critical_ds.add(max_d + 1)
        critical_ds = sorted(critical_ds)
        MOD = 10**9 + 7
        sum_total = 0
        for i in range(1, len(critical_ds)):
            d_low = critical_ds[i-1]
            d_high = critical_ds[i]
            if d_low < 1:
                continue
            count = max(0, d_high - d_low)
            fd = self.compute_f(nums, k, d_low)
            sum_total = (sum_total + fd * count) % MOD
        return sum_total % MOD
    
    def compute_f(self, s: List[int], k: int, d: int) -> int:
        n = len(s)
        dp = [[0] * (k + 1) for _ in range(n)]
        for i in range(n):
            dp[i][1] = 1
        for j in range(2, k + 1):
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i + 1] = prefix[i] + dp[i][j - 1]
            for i in range(n):
                low, high = 0, i - 1
                res = -1
                while low <= high:
                    mid = (low + high) // 2
                    if s[i] - s[mid] >= d:
                        res = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                dp[i][j] = prefix[res + 1] if res != -1 else 0
        total = 0
        for i in range(n):
            total += dp[i][k]
        return total