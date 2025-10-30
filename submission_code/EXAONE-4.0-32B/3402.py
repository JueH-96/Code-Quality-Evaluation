import math

class Solution:
    def minCostToEqualizeArray(self, nums: list[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 1:
            return 0
        
        max_val = max(nums)
        min_val = min(nums)
        total_sum = sum(nums)
        
        if cost2 >= 2 * cost1:
            T = max_val
            s_val = n * T - total_sum
            cost = s_val * cost1
            return cost % MOD
        
        else:
            if n == 2:
                T = max_val
                s_val = n * T - total_sum
                M_val = T - min_val
                k_max = min(s_val // 2, s_val - M_val)
                cost = s_val * cost1 + k_max * (cost2 - 2 * cost1)
                return cost % MOD
            
            else:
                candidates = {max_val}
                numerator = total_sum - 2 * min_val
                denominator = n - 2
                T0 = numerator / denominator
                
                if T0 >= max_val:
                    t1 = math.floor(T0)
                    t2 = math.ceil(T0)
                    if t1 >= max_val:
                        candidates.add(t1)
                    if t2 >= max_val:
                        candidates.add(t2)
                else:
                    candidates.add(max_val + 1)
                
                best = None
                for T in sorted(candidates):
                    s_val = n * T - total_sum
                    M_val = T - min_val
                    k_max = min(s_val // 2, s_val - M_val)
                    cost_val = s_val * cost1 + k_max * (cost2 - 2 * cost1)
                    if best is None or cost_val < best:
                        best = cost_val
                
                return best % MOD