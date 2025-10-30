from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def prime_factors(n):
            factors = defaultdict(int)
            # Count the number of 2s that divide n
            while n % 2 == 0:
                factors[2] += 1
                n //= 2
            # n must be odd at this point, so a skip of 2 ( i = i + 2) can be used
            for i in range(3, int(n**0.5)+1, 2):
                # while i divides n, append i and divide n
                while n % i == 0:
                    factors[i] += 1
                    n //= i
            # Condition if n is a prime number greater than 2
            if n > 2:
                factors[n] += 1
            return factors
        
        def normalize_factors(factors):
            return {x: factors[x] % 2 for x in factors if factors[x] % 2 != 0}
        
        # Create a dictionary to store the maximum sum for each unique factorization
        max_sum = defaultdict(int)
        
        for num in nums:
            factors = prime_factors(num)
            normalized = normalize_factors(factors)
            # Convert the dictionary to a frozenset to use it as a key
            key = frozenset(normalized.items())
            max_sum[key] = max(max_sum[key], max_sum[key] + num)
        
        # Return the maximum sum of all the unique factorizations
        return max(max_sum.values())

# Example usage:
# sol = Solution()
# print(sol.maximumSum([8,7,3,5,7,2,4,9]))  # Output: 16
# print(sol.maximumSum([5,10,3,10,1,13,7,9,4]))  # Output: 19