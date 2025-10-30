from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        
        # Define the set of prime numbers up to 100.
        # Numbers in `nums` are constrained to be between 1 and 100.
        # Note: 1 is not a prime number.
        PRIMES_UP_TO_100 = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        }

        min_prime_idx = -1  # Initialize to an invalid index
        # This variable will be updated to store the index of the last prime number encountered.
        max_prime_idx = -1  # Initialize to an invalid index

        # Iterate through the input list `nums` along with their indices.
        for i, num in enumerate(nums):
            # Check if the current number is prime using the precomputed set.
            if num in PRIMES_UP_TO_100:
                if min_prime_idx == -1:
                    # If this is the first prime number we've encountered,
                    # record its index in min_prime_idx.
                    min_prime_idx = i
                
                # Always update max_prime_idx to the index of the current prime number.
                # After the loop finishes, max_prime_idx will hold the index of the
                # rightmost prime number in the list.
                max_prime_idx = i
            
        # According to the problem constraints:
        # "The input is generated such that the number of prime numbers in the nums is at least one."
        # This guarantee ensures that `min_prime_idx` will be updated to a valid non-negative index.
        # Consequently, `max_prime_idx` will also be updated to a valid non-negative index,
        # and `max_prime_idx` will be greater than or equal to `min_prime_idx`.
        
        # If there is only one prime number in the list (e.g., nums = [4, 8, 2, 8]),
        # then `min_prime_idx` and `max_prime_idx` will both be set to the index of that single prime.
        # In this case, `max_prime_idx - min_prime_idx` will be 0, which is the correct output.
        # (e.g., for [4,8,2,8], prime is 2 at index 2. min_idx=2, max_idx=2. Difference is 0).

        # The maximum distance is the difference between the index of the last prime
        # and the index of the first prime.
        return max_prime_idx - min_prime_idx