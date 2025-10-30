class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        P = [0] * (n + 1)
        for i in range(1, n + 1):
            P[i] = P[i - 1] + nums[i - 1]
        
        mult = [0] * (k + 1)
        for j in range(1, k + 1):
            sign = 1 if j % 2 == 1 else -1
            mult[j] = (k - j + 1) * sign
        
        dp_prev = [0] * (n + 1)
        NEG_INF = -10**18
        
        for j in range(1, k + 1):
            dp_curr = [NEG_INF] * (n + 1)
            best_val = 0
            mult_j = mult[j]
            
            for i in range(1, n + 1):
                candidate = mult_j * P[i] + best_val
                if i == 1:
                    dp_curr[i] = candidate
                else:
                    dp_curr[i] = max(dp_curr[i - 1], candidate)
                
                current_val = dp_prev[i] - mult_j * P[i]
                if current_val > best_val:
                    best_val = current_val
            
            dp_prev = dp_curr
        
        return dp_prev[n]