from math import gcd
from heapq import nlargest
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        """
        Finds the subset with the maximum sum from the given array nums,
        such that the product of any two elements in the subset is a perfect square.
        """
        def prime_factors(num):
            """
            Returns the prime factors of a given number as a dictionary.
            """
            factors = defaultdict(int)
            while num % 2 == 0:
                factors[2] += 1
                num //= 2
            cur_prime = 3
            while cur_prime * cur_prime <= num:
                while num % cur_prime == 0:
                    factors[cur_prime] += 1
                    num //= cur_prime
                cur_prime += 2
            if num != 1:
                factors[num] += 1
            return factors
        
        # Storing prime factors along with their indices
       FactorMap = []
        for i, num in enumerate(nums):
            FactorMap.append((prime_factors(num), i))
        
        # Storing indices associated with the same prime factors after sorting by prime factors
        HashMap = defaultdict(list)
        for (factors, i) in FactorMap:
            HashMap[tuple(factors.items())].append(i)
        
        result = 0
        
        for indices in HashMap.values():
            if len(indices) > 1:
                temp_nums = [nums[i] for i in indices]
                # Using a max heap to get the three largest elements for each group
                temp_nums = nlargest(3, temp_nums)
                # Caclulate the sum of the largest three numbers to get the maximum possible sum
                result = max(result, sum(temp_nums))
        
        # Checking the sum of each individual number, since a set with only one element is inherently 'complete'
        result = max(result, max(nums))
        
        return result