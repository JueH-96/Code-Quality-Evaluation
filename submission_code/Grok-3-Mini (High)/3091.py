import math
from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq_counter = Counter(nums)
        f_zero = freq_counter.get(0, 0)
        if 0 in freq_counter:
            del freq_counter[0]
        total_sum_non_zero = 0
        for val, cnt in freq_counter.items():
            total_sum_non_zero += val * cnt
        dp = [0] * (total_sum_non_zero + 1)
        dp[0] = 1
        for val, max_count in freq_counter.items():
            new_dp = [0] * (total_sum_non_zero + 1)
            v = val
            dp_r_lists = [[] for _ in range(v)]
            for s_idx in range(total_sum_non_zero + 1):
                residue = s_idx % v
                dp_r_lists[residue].append(dp[s_idx])
            prefix_lists = []
            for r in range(v):
                a_r = dp_r_lists[r]
                prefix_r = [0]
                cum_sum = 0
                for num in a_r:
                    cum_sum += num
                    cum_sum %= MOD
                    prefix_r.append(cum_sum)
                prefix_lists.append(prefix_r)
            for s in range(total_sum_non_zero + 1):
                r_s = s % v
                prefix_r = prefix_lists[r_s]
                q_s = s // v
                m = min(max_count, q_s)
                low_idx = q_s - m
                high_idx = q_s
                sum_k = (prefix_r[high_idx + 1] - prefix_r[low_idx] + MOD) % MOD
                new_dp[s] = sum_k
            dp = new_dp
        low = max(0, l)
        high = min(r, total_sum_non_zero)
        if low > high:
            sum_range = 0
        else:
            sum_range = 0
            for s_sum in range(low, high + 1):
                sum_range += dp[s_sum]
                sum_range %= MOD
        answer = (sum_range * (f_zero + 1) % MOD) % MOD
        return answer