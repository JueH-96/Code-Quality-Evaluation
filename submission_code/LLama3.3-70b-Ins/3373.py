from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Function to check if a number is prime
        def is_prime(n: int) -> bool:
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        # Find the indices of all prime numbers in the array
        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]
        
        # Calculate the maximum distance between the indices of two prime numbers
        max_distance = max(prime_indices) - min(prime_indices)
        
        return max_distance