import math
from typing import List # Ensure List is imported if needed by the environment

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        """
        Calculates the maximum distance between the indices of two prime numbers in the list.

        The distance between two indices i and j is |i - j|. We want to find the maximum
        such distance where both nums[i] and nums[j] are prime numbers. This maximum
        distance will occur between the first prime number found in the list and the
        last prime number found in the list.

        Args:
            nums: A list of integers where each element is between 1 and 100.
                  It's guaranteed that at least one prime number exists in the list.

        Returns:
            The maximum difference between the indices of the first and last prime numbers found.
        """
        
        # Set of prime numbers between 1 and 100 (inclusive).
        # Numbers <= 1 are not prime. 
        # The primes up to 100 are:
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

        min_prime_idx = -1  # Initialize index of the first prime found. Set to -1 initially.
        max_prime_idx = -1  # Initialize index of the last prime found. Set to -1 initially.

        # Iterate through the list with index `i` and value `num`.
        for i, num in enumerate(nums):
            # Check if the current number `num` is prime using the precomputed set.
            # Set lookup is efficient (average O(1) time).
            if num in primes:
                # If this is the first prime number we've encountered (min_prime_idx is still -1),
                # record its index `i` in min_prime_idx.
                if min_prime_idx == -1:
                    min_prime_idx = i
                
                # Always update max_prime_idx to the index `i` of the current prime number.
                # This ensures that after iterating through the entire list, 
                # max_prime_idx will hold the index of the very last prime number found.
                max_prime_idx = i

        # The problem statement guarantees that there is at least one prime number in nums.
        # Therefore, after the loop, min_prime_idx will have been assigned a valid non-negative index.
        # max_prime_idx will also hold a valid index (it will be >= min_prime_idx).
        # If there's only one prime number in the list, min_prime_idx and max_prime_idx will be the same.
        # The maximum distance is the difference between the index of the last prime found
        # and the index of the first prime found.
        return max_prime_idx - min_prime_idx