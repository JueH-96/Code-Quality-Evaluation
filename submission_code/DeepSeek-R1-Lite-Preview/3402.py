class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        t = max(nums)
        d = [t - num for num in nums]
        S = sum(d)
        m = sum(1 for di in d if di >= 1)
        
        if cost2 < 2 * cost1 and m >= 2:
            k_max = S // 2
            cost = (k_max * cost2) % MOD
            cost = (cost + (S - 2 * k_max) * cost1) % MOD
        else:
            cost = (S * cost1) % MOD
        
        return cost