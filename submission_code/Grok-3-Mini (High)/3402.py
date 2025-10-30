from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 1000000007
        max_val = max(nums)
        min_num = min(nums)
        sum_nums = sum(nums)
        len_nums = len(nums)
        sum_delta = len_nums * max_val - sum_nums
        max_delta = max_val - min_num
        max_E = min(sum_delta // 2, sum_delta - max_delta)
        C = cost2 - 2 * cost1
        if C < 0:
            E = max_E
        elif C > 0:
            E = 0
        else:
            E = 0  # Arbitrary choice when C == 0
        cost = (cost1 * sum_delta + C * E) % MOD
        return cost