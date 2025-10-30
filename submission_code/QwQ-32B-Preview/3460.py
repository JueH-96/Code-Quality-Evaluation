from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        requirements.sort(key=lambda x: x[0])
        req_idx = 0
        req_end, req_cnt = requirements[0]
        
        # Initialize DP: dp[pos][inv]
        dp = [0] * 401  # Adjust size based on maximum inversion count
        dp[0] = 1  # Base case: one way to have 0 inversions with 0 elements
        
        used = [False] * n
        fact = [1] * n  # Precompute factorials for combinations
        
        for i in range(1, n):
            fact[i] = fact[i - 1] * i % MOD
        
        for pos in range(n):
            if pos > 0:
                # Shift the DP to the next position
                new_dp = [0] * 401
                for inv in range(401):
                    if dp[inv] > 0:
                        remaining = n - pos
                        for x in range(n):
                            if not used[x]:
                                # Calculate new inversions introduced by x
                                new_invs = inv + sum(1 for y in range(pos + 1, n) if not used[y] and y < x)
                                if new_invs < 401:
                                    new_dp[new_invs] = (new_dp[new_invs] + dp[inv]) % MOD
                dp = new_dp
            if pos == req_end:
                # Check if the current inversion count matches the requirement
                if dp[req_cnt] == 0:
                    return 0
                # Move to the next requirement
                req_idx += 1
                if req_idx < len(requirements):
                    req_end, req_cnt = requirements[req_idx]
        
        return dp[req_cnt]