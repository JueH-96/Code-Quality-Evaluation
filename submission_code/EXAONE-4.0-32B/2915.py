from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        P = [0] * (n + 1)
        for i in range(n):
            if nums[i] % modulo == k:
                P[i + 1] = P[i] + 1
            else:
                P[i + 1] = P[i]
        
        Q = [p % modulo for p in P]
        
        freq = defaultdict(int)
        freq[Q[0]] = 1
        ans = 0
        
        for i in range(1, n + 1):
            residue = Q[i]
            target = (residue - k) % modulo
            ans += freq.get(target, 0)
            freq[residue] += 1
            
        return ans