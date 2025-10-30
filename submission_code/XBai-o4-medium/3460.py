from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        required = [None] * (n + 1)  # m ranges from 1 to n
        for end, cnt in requirements:
            m_val = end + 1
            required[m_val] = cnt
        
        # Initialize for m = 1
        prev_dp = [0] * 1
        prev_dp[0] = 1
        
        for m in range(2, n + 1):
            prev_max_c = (m - 1) * (m - 2) // 2
            current_max_c = m * (m - 1) // 2
            current_dp = [0] * (current_max_c + 1)
            
            # Compute prefix sums of prev_dp
            prefix_sum = [0] * (len(prev_dp) + 1)
            for i in range(len(prev_dp)):
                prefix_sum[i + 1] = (prefix_sum[i] + prev_dp[i]) % MOD
            
            # Fill current_dp
            for c_new in range(current_max_c + 1):
                a = max(0, c_new - (m - 1))
                b = min(c_new, prev_max_c)
                if a > b:
                    current_dp[c_new] = 0
                else:
                    sum_val = (prefix_sum[b + 1] - prefix_sum[a]) % MOD
                    current_dp[c_new] = sum_val
            
            # Apply requirement for this m
            if required[m] is not None:
                req = required[m]
                if req < 0 or req > current_max_c:
                    return 0
                # Mask current_dp to only req
                temp = current_dp[req]
                current_dp = [0] * (current_max_c + 1)
                current_dp[req] = temp
            
            prev_dp = current_dp
        
        return prev_dp[required[n]] % MOD