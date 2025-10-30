from typing import List
import math
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_g = max(nums)
        res = 0
        
        for g in range(1, max_g + 1):
            S_g = [x for x in nums if x % g == 0]
            n = len(S_g)
            if n < 2:
                continue
            T_g = [x // g for x in S_g]
            
            dp = defaultdict(int)
            dp[(0, 0, False, False)] = 1
            
            for x in T_g:
                new_dp = defaultdict(int)
                for state in dp:
                    a_prev, b_prev, hasA_prev, hasB_prev = state
                    cnt = dp[state]
                    
                    new_dp[state] = (new_dp[state] + cnt) % MOD
                    
                    new_a = math.gcd(a_prev, x) if hasA_prev else x
                    new_state_A = (new_a, b_prev, True, hasB_prev)
                    new_dp[new_state_A] = (new_dp[new_state_A] + cnt) % MOD
                    
                    new_b = math.gcd(b_prev, x) if hasB_prev else x
                    new_state_B = (a_prev, new_b, hasA_prev, True)
                    new_dp[new_state_B] = (new_dp[new_state_B] + cnt) % MOD
                
                dp = new_dp
            
            total = 0
            for state in dp:
                a, b, hasA, hasB = state
                if hasA and hasB and a == 1 and b == 1:
                    total = (total + dp[state]) % MOD
            res = (res + total) % MOD
        
        return res