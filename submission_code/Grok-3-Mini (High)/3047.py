import math
from collections import defaultdict
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Compute primes up to 100 using Sieve of Eratosthenes
        is_prime = [True] * 101
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.sqrt(100)) + 1):  # Range from 2 to 10
            if is_prime[i]:
                for j in range(i * i, 101, i):
                    if j < 101:
                        is_prime[j] = False
        primes = [p for p in range(2, 101) if is_prime[p]]
        
        # Define a helper function to compute the square-free kernel of a number
        def square_free_kernel(num):
            k = num
            max_p_check = int(math.sqrt(num))
            for p in primes:
                if p > max_p_check:
                    break
                if k % (p * p) == 0:
                    while k % (p * p) == 0:
                        k //= (p * p)
            return k
        
        # Use a dictionary to sum nums values grouped by square-free kernel of indices
        sum_dict = defaultdict(int)
        for i in range(1, n + 1):  # Indices from 1 to n (1-based)
            kernel = square_free_kernel(i)
            sum_dict[kernel] += nums[i - 1]  # nums is 0-based, so access nums[i-1]
        
        # The maximum sum is the maximum value in the dictionary
        return max(sum_dict.values())