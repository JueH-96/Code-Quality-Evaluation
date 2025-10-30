from typing import List
from itertools import permutations
from math import factorial

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Generate all permutations of nums
        perms = list(permutations(nums))
        
        # Initialize count of special permutations
        count = 0
        
        # Iterate over each permutation
        for perm in perms:
            # Assume the permutation is special
            is_special = True
            
            # Check if the permutation is special
            for i in range(n - 1):
                if perm[i] % perm[i+1] != 0 and perm[i+1] % perm[i] != 0:
                    # If not special, break the loop
                    is_special = False
                    break
            
            # If the permutation is special, increment the count
            if is_special:
                count += 1
        
        # Return the count modulo MOD
        return count % MOD