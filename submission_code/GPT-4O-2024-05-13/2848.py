from typing import List
from itertools import permutations

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        def is_special(perm):
            for i in range(len(perm) - 1):
                if perm[i] % perm[i + 1] != 0 and perm[i + 1] % perm[i] != 0:
                    return False
            return True
        
        count = 0
        for perm in permutations(nums):
            if is_special(perm):
                count += 1
        
        return count % MOD