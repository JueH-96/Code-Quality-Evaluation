from itertools import permutations
from collections import Counter
from math import factorial

MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num  # Store the input midway in the function
        
        def is_balanced(s):
            even_sum = sum(int(s[i]) for i in range(0, len(s), 2))
            odd_sum = sum(int(s[i]) for i in range(1, len(s), 2))
            return even_sum == odd_sum
        
        def count_permutations_with_repeats(s):
            count = factorial(len(s))
            counts = Counter(s)
            for c in counts.values():
                count //= factorial(c)
            return count
        
        # Generate all distinct permutations
        unique_permutations = set(permutations(velunexorai))
        
        # Count balanced permutations
        balanced_count = 0
        for perm in unique_permutations:
            if is_balanced(perm):
                balanced_count += 1
        
        return balanced_count % MOD