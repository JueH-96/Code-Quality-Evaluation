from typing import List

class Solution:
    # Pre-compute all prime numbers up to 100.
    # This set will be used for O(1) average time complexity lookups.
    _primes_up_to_100 = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 
        59, 61, 67, 71, 73, 79, 83, 89, 97
    }

    def maximumPrimeDifference(self, nums: List[int]) -> int:
        first_prime_idx = -1
        last_prime_idx = -1

        # Iterate through the array with indices
        for i, num in enumerate(nums):
            # Check if the current number is prime using the pre-computed set
            if num in self._primes_up_to_100:
                # If this is the first prime number found, record its index
                if first_prime_idx == -1:
                    first_prime_idx = i
                # Always update the last prime index to the current prime's index
                # This ensures last_prime_idx holds the rightmost prime's index
                last_prime_idx = i
        
        # The maximum difference is the distance between the first and last prime found
        return last_prime_idx - first_prime_idx