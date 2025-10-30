class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        req = {}
        for end, cnt in requirements:
            m = end + 1
            req[m] = cnt
        
        prev_dp = [1]  # Initial state for m=0 (0 inversions)
        
        for m in range(1, n + 1):
            max_inv = m * (m - 1) // 2
            curr_dp = [0] * (max_inv + 1)
            prev_max_inv = (m - 1) * (m - 2) // 2 if m >= 2 else 0
            
            for k_prev in range(len(prev_dp)):
                if prev_dp[k_prev] == 0:
                    continue
                for i in range(m):
                    added = (m - 1) - i
                    new_k = k_prev + added
                    if new_k <= max_inv:
                        curr_dp[new_k] = (curr_dp[new_k] + prev_dp[k_prev]) % MOD
            
            if m in req:
                required = req[m]
                if required > max_inv:
                    return 0
                for k in range(len(curr_dp)):
                    if k != required:
                        curr_dp[k] = 0
                if curr_dp[required] == 0:
                    return 0
            
            prev_dp = curr_dp
        
        return prev_dp[req[n]] % MOD