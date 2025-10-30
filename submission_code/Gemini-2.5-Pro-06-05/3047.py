import collections
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        """
        Calculates the maximum element-sum of a complete subset of indices.

        A subset of indices is "complete" if the product of every pair of indices is a perfect square.
        This condition holds if and only if all indices in the subset share the same "square-free part".
        The square-free part of a number is what remains after dividing out all its perfect square factors.

        The algorithm proceeds as follows:
        1. Partition indices {1, ..., n} into groups based on their square-free part.
        2. For each group, calculate the sum of the corresponding values from the `nums` array.
        3. The maximum of these sums is the answer.

        To efficiently find the square-free part of each index `i`, we first use a sieve to
        precompute the Smallest Prime Factor (SPF) for all numbers up to `n`. Then, for each `i`,
        we find its prime factorization using the SPF array to construct its square-free part.
        """
        n = len(nums)

        # Part 1: Sieve to find the smallest prime factor (spf) for each number up to n.
        # Time complexity: O(n log log n)
        # Space complexity: O(n)
        spf = list(range(n + 1))
        # We only need to check for primes up to sqrt(n)
        for i in range(2, int(n**0.5) + 1):
            if spf[i] == i:  # This condition means i is a prime number
                # Mark the smallest prime factor for all multiples of i
                for j in range(i * i, n + 1, i):
                    if spf[j] == j:  # Only update if it's not already marked by a smaller prime
                        spf[j] = i

        # Part 2: Group elements by the square-free part of their 1-based index.
        group_sums = collections.defaultdict(int)
        
        # Iterate through indices from 1 to n.
        # Time complexity for this loop: O(n * log n)
        for i in range(1, n + 1):
            # Calculate the square-free core of the index 'i'
            # by finding its prime factorization. The core is the product
            # of prime factors with an odd exponent.
            num = i
            core = 1
            while num > 1:
                prime_factor = spf[num]
                count = 0
                while num % prime_factor == 0:
                    count += 1
                    num //= prime_factor
                
                if count % 2 == 1:
                    core *= prime_factor
            
            # The problem uses 1-based indexing, so we access nums with i - 1.
            group_sums[core] += nums[i - 1]

        # Part 3: Find the maximum sum among all groups.
        # Since n >= 1, group_sums will not be empty.
        return max(group_sums.values())