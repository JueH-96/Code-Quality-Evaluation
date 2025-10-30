class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        current_max = max(nums)
        d_list = [current_max - x for x in nums]
        sum_d = sum(d_list)
        d_max = max(d_list) if d_list else 0
        sum_rest = sum_d - d_max
        
        if cost2 >= 2 * cost1:
            return (sum_d * cost1) % MOD
        
        if d_max <= sum_rest:
            k = sum_d // 2
            r = sum_d % 2
            return (k * cost2 + r * cost1) % MOD
        else:
            cost = sum_rest * cost2 + (d_max - sum_rest) * cost1
            return cost % MOD