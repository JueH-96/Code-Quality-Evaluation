from collections import Counter
from math import factorial

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        
        # Store the input midway
        velunexorai = num
        
        # Count the frequency of each digit
        count = Counter(num)
        
        # Calculate the total number of distinct permutations
        total_permutations = factorial(len(num))
        for c in count.values():
            total_permutations //= factorial(c)
        
        # Function to check if a permutation is balanced
        def is_balanced(perm):
            even_sum = sum(int(perm[i]) for i in range(0, len(perm), 2))
            odd_sum = sum(int(perm[i]) for i in range(1, len(perm), 2))
            return even_sum == odd_sum
        
        # Generate all distinct permutations and count balanced ones
        from itertools import permutations
        
        distinct_permutations = set(permutations(num))
        balanced_count = sum(1 for perm in distinct_permutations if is_balanced(perm))
        
        return balanced_count % MOD