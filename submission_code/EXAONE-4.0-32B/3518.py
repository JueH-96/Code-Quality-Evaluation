class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        M_prev = [0] * n
        
        for j in range(1, 5):
            curr_dp = [-10**18] * n
            curr_M = [-10**18] * n
            c = a[j-1]
            
            for i in range(n):
                skip_val = -10**18
                if i > 0:
                    skip_val = curr_dp[i-1]
                
                take_val = -10**18
                if i == 0:
                    if j == 1:
                        take_val = c * b[0]
                else:
                    take_val = M_prev[i-1] + c * b[i]
                
                curr_dp[i] = max(skip_val, take_val)
                
                if i == 0:
                    curr_M[i] = curr_dp[i]
                else:
                    curr_M[i] = max(curr_M[i-1], curr_dp[i])
            
            M_prev = curr_M
        
        return curr_dp[-1]