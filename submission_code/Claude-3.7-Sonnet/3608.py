from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # Maps GCD value -> list of bitmasks with that GCD
        gcd_to_bitmasks = defaultdict(list)
        
        # Generate all non-empty subsequences and compute their GCDs
        for mask in range(1, 1 << n):
            g = 0
            for i in range(n):
                if mask & (1 << i):
                    g = gcd(g, nums[i]) if g else nums[i]
            gcd_to_bitmasks[g].append(mask)
        
        count = 0
        # Count disjoint pairs for each GCD value
        for g, bitmasks in gcd_to_bitmasks.items():
            m = len(bitmasks)
            for i in range(m):
                for j in range(i+1, m):
                    if (bitmasks[i] & bitmasks[j]) == 0:
                        # Count both (i,j) and (j,i) as different pairs
                        count = (count + 2) % MOD
        
        return count