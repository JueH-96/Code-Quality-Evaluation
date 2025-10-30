from typing import List
from math import gcd
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        max_gcd = 200
        total = 0
        
        for g in range(1, max_gcd + 1):
            # Filter elements divisible by g and divide by g
            S = [x // g for x in nums if x % g == 0]
            m = len(S)
            if m < 2:
                continue  # Need at least two elements to form two non-empty subsequences
            
            # Initialize DP: (a, b) represents the GCD of the first and second subsequences
            dp = defaultdict(int)
            dp[(0, 0)] = 1  # Initial state: both subsequences are empty
            
            for s in S:
                new_dp = defaultdict(int)
                for (a, b), cnt in dp.items():
                    # Option 1: Add s to the first subsequence
                    new_a = gcd(a, s) if a != 0 else s
                    new_b = b
                    new_dp[(new_a, new_b)] = (new_dp[(new_a, new_b)] + cnt) % MOD
                    
                    # Option 2: Add s to the second subsequence
                    new_a2 = a
                    new_b2 = gcd(b, s) if b != 0 else s
                    new_dp[(new_a2, new_b2)] = (new_dp[(new_a2, new_b2)] + cnt) % MOD
                    
                    # Option 3: Do not add s to either subsequence
                    new_dp[(a, b)] = (new_dp[(a, b)] + cnt) % MOD
                dp = new_dp
            
            # The valid pairs are those where both subsequences have GCD 1 (after division by g)
            total = (total + dp.get((1, 1), 0)) % MOD
        
        return total