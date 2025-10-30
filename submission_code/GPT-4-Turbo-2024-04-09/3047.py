class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        from math import gcd
        from functools import reduce
        from sympy import factorint
        
        def prime_factors(x):
            return factorint(x)
        
        def normalize_factors(factors):
            return {p: exp % 2 for p, exp in factors.items() if exp % 2 == 1}
        
        def tuple_key(factors):
            return tuple(sorted(factors.items()))
        
        # Map each number to its normalized prime factors with odd exponents
        factor_map = {}
        for num in nums:
            factors = prime_factors(num)
            normalized = normalize_factors(factors)
            factor_map[num] = normalized
        
        # Group numbers by their normalized factor keys
        groups = defaultdict(list)
        for num, factors in factor_map.items():
            key = tuple_key(factors)
            groups[key].append(num)
        
        # Find the maximum sum of a complete subset
        max_sum = 0
        for group in groups.values():
            max_sum = max(max_sum, sum(group))
        
        return max_sum