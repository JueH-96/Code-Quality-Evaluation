from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        """
        Finds the maximum distance between the indices of two prime numbers in the array.
        Iterates through the array once to find the index of the first and last prime.
        The maximum distance is the difference between these indices.
        
        Args:
            nums: An integer array. Contains at least one prime number.
                  1 <= nums[i] <= 100.
        
        Returns:
            The maximum distance between the indices of two prime numbers.
        """
        # Define the set of prime numbers up to the maximum possible value in nums (100).
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        
        first_prime_index = -1
        last_prime_index = -1
        
        # Iterate through the array to find the index of the first occurrence of a prime
        # and keep track of the index of the last occurrence of a prime.
        for i in range(len(nums)):
            if nums[i] in primes:
                # If this is the first prime found so far, record its index.
                if first_prime_index == -1:
                    first_prime_index = i
                
                # Always update the index of the last prime found so far.
                last_prime_index = i
                
        # The problem guarantees that there is at least one prime number in nums.
        # Therefore, both first_prime_index will be updated from -1 to the index of the first prime,
        # and last_prime_index will be updated at least once, ending up with the index of the last prime.
        # The last_prime_index will always be greater than or equal to first_prime_index.
        # The maximum distance between any two prime indices is the distance between
        # the first prime's index and the last prime's index.
        return last_prime_index - first_prime_index