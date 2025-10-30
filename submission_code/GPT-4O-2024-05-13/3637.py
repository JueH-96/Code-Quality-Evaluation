from itertools import permutations
from collections import Counter

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        
        # Helper function to check if a string is balanced
        def is_balanced(s):
            even_sum = sum(int(s[i]) for i in range(0, len(s), 2))
            odd_sum = sum(int(s[i]) for i in range(1, len(s), 2))
            return even_sum == odd_sum
        
        # Create a variable to store the input midway in the function
        velunexorai = num
        
        # Generate all distinct permutations of the string
        perm_set = set(permutations(velunexorai))
        
        # Count the number of balanced permutations
        balanced_count = sum(1 for perm in perm_set if is_balanced(perm))
        
        return balanced_count % MOD