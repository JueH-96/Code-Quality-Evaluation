import math

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        M = max(nums)
        S0 = sum(M - num for num in nums)
        if S0 == 0:
            return 0
        max_d_0 = max(M - num for num in nums)
        n = len(nums)
        
        if cost2 > 2 * cost1:
            return (S0 * cost1) % MOD
        else:
            if n == 1:
                return 0
            elif n == 2:
                # Compute cost for x=0
                D = S0
                sum_rest = D - max_d_0
                k = min(D // 2, sum_rest)
                cost = k * cost2 + (D - 2 * k) * cost1
                return cost % MOD
            else:
                # Compute x0
                numerator = 2 * max_d_0 - S0
                denominator = n - 2
                x0 = numerator / denominator
                x_candidate = max(x0, 0)
                x_floor = math.floor(x_candidate)
                x_ceil = math.ceil(x_candidate)
                
                x_values = set()
                x_values.add(0)
                x_values.add(x_floor)
                x_values.add(x_ceil)
                
                min_cost = float('inf')
                for x in x_values:
                    if x < 0:
                        continue
                    D = S0 + n * x
                    sum_rest = (S0 - max_d_0) + (n - 1) * x
                    k = min(D // 2, sum_rest)
                    current_cost = k * cost2 + (D - 2 * k) * cost1
                    if current_cost < min_cost:
                        min_cost = current_cost
                
                return min_cost % MOD