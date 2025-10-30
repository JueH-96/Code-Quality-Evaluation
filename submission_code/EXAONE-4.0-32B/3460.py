MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        req_dict = {}
        for end, cnt in requirements:
            req_dict[end] = cnt
        
        current_min = 0
        current_max = 0
        dp = [1]
        
        for i in range(n):
            if i in req_dict:
                c_required = req_dict[i]
                if c_required < current_min or c_required > current_max:
                    dp = [0]
                    current_min = c_required
                    current_max = c_required
                else:
                    idx = c_required - current_min
                    val = dp[idx]
                    dp = [val]
                    current_min = c_required
                    current_max = c_required
            
            if i == n - 1:
                break
            
            next_min = current_min
            next_max = current_max + (i + 1)
            L_next = next_max - next_min + 1
            next_dp = [0] * L_next
            
            if dp:
                P = [0] * len(dp)
                P[0] = dp[0] % MOD
                for j in range(1, len(dp)):
                    P[j] = (P[j - 1] + dp[j]) % MOD
                
                for x in range(next_min, next_max + 1):
                    low_j = max(current_min, x - (i + 1))
                    high_j = min(current_max, x)
                    
                    if low_j > high_j:
                        s = 0
                    else:
                        idx_low = low_j - current_min
                        idx_high = high_j - current_min
                        if idx_low == 0:
                            s = P[idx_high]
                        else:
                            s = (P[idx_high] - P[idx_low - 1]) % MOD
                        if s < 0:
                            s += MOD
                    pos_in_next = x - next_min
                    next_dp[pos_in_next] = s % MOD
            
            dp = next_dp
            current_min = next_min
            current_max = next_max
        
        return dp[0] % MOD