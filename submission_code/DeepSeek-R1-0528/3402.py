class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        M0 = max(nums)
        A = min(nums)
        B = sum(nums)
        
        candidate_T = {M0}
        
        if n > 2:
            numerator = B - 2 * A
            denominator = n - 2
            floor_val = numerator // denominator
            if numerator % denominator == 0:
                ceil_val = floor_val
            else:
                ceil_val = floor_val + 1
            if floor_val >= M0:
                candidate_T.add(floor_val)
            if ceil_val >= M0:
                candidate_T.add(ceil_val)
        
        min_cost = float('inf')
        
        for T in candidate_T:
            S = n * T - B
            max_d = T - A
            
            if 2 * max_d > S:
                rem = 2 * max_d - S
                R = S - rem
                if cost2 < 2 * cost1:
                    term = (n - 1) * T - (B - A)
                    cost_val = rem * cost1 + term * cost2
                else:
                    cost_val = S * cost1
            else:
                if cost2 < 2 * cost1:
                    half = S // 2
                    remainder = S % 2
                    cost_val = half * cost2 + remainder * cost1
                else:
                    cost_val = S * cost1
            
            if cost_val < min_cost:
                min_cost = cost_val
        
        return min_cost % mod