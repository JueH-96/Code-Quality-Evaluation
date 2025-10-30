import bisect

mod = 10**9 + 7

class Solution:
    def sumOfPowers(self, nums: list, k: int) -> int:
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        max_gap = sorted_nums[-1] - sorted_nums[0]
        
        T_original = set()
        for i in range(n):
            for j in range(i + 1, n):
                diff = sorted_nums[j] - sorted_nums[i]
                T_original.add(diff)
        
        T_sorted = set()
        T_sorted.add(0)
        T_sorted.add(max_gap + 1)
        for g in T_original:
            T_sorted.add(g)
            T_sorted.add(g + 1)
        T_sorted = sorted(T_sorted)
        
        F_dict = {}
        for g_val in T_sorted:
            dp = [1] * n
            for l in range(2, k + 1):
                new_dp = [0] * n
                prefix = [0] * n
                prefix[0] = dp[0]
                for i in range(1, n):
                    prefix[i] = prefix[i - 1] + dp[i]
                for i in range(n):
                    x = sorted_nums[i] - g_val
                    idx = bisect.bisect_right(sorted_nums, x) - 1
                    if idx < 0:
                        new_dp[i] = 0
                    else:
                        if idx >= i:
                            idx = i - 1
                        new_dp[i] = prefix[idx]
                dp = new_dp
            total = sum(dp)
            F_dict[g_val] = total
        
        ans = 0
        for g_val in T_original:
            term = g_val * (F_dict[g_val] - F_dict[g_val + 1])
            ans = (ans + term) % mod
        return ans