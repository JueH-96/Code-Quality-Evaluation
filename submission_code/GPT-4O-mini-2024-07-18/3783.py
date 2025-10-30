from itertools import permutations
from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Generate the list of the first n positive integers
        nums = list(range(1, n + 1))
        
        # Generate all permutations of the list
        all_perms = permutations(nums)
        
        # Filter for alternating permutations
        alternating_perms = []
        for perm in all_perms:
            valid = True
            for i in range(1, n):
                if (perm[i] % 2 == perm[i - 1] % 2):  # Check if both are odd or both are even
                    valid = False
                    break
            if valid:
                alternating_perms.append(perm)
        
        # Sort the valid permutations lexicographically
        alternating_perms.sort()
        
        # Check if k is within the range of valid permutations
        if k > len(alternating_perms):
            return []
        
        # Return the k-th permutation (1-indexed)
        return list(alternating_perms[k - 1])