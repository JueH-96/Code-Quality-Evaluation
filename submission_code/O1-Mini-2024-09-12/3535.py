from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        max_num = max(nums)
        dp_prev = [0] * (max_num + 1)
        for a in range(nums[0] + 1):
            dp_prev[a] = 1
        for i in range(1, n):
            dp_curr = [0] * (max_num + 1)
            delta = nums[i] - nums[i - 1]
            for a_prev in range(nums[i - 1] + 1):
                if dp_prev[a_prev] == 0:
                    continue
                if delta > 0:
                    lb = a_prev + delta
                else:
                    lb = a_prev
                if lb > nums[i]:
                    continue
                for a in range(lb, nums[i] + 1):
                    dp_curr[a] = (dp_curr[a] + dp_prev[a_prev]) % MOD
            dp_prev = dp_curr
        return sum(dp_prev[:nums[-1] + 1]) % MOD