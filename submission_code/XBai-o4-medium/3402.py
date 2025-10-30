class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 1:
            return 0
        max_num = max(nums)
        min_num = min(nums)
        sum_nums = sum(nums)
        # Calculate total deficit when target is max_num
        sum_d = n * max_num - sum_nums
        if cost2 >= 2 * cost1:
            return (sum_d * cost1) % MOD
        else:
            max_d = max_num - min_num
            sum_d_minus_max_d = sum_d - max_d
            sum_d_over_2 = sum_d // 2
            max_ops = min(sum_d_minus_max_d, sum_d_over_2)
            total_cost = (max_ops * cost2) + ((sum_d - 2 * max_ops) * cost1)
            return total_cost % MOD