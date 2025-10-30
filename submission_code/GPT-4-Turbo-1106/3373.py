from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Helper function to check if a number is prime
        def is_prime(n):
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
        
        # Find indices of prime numbers
        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]
        
        # Return the maximum distance between prime indices
        return max(prime_indices) - min(prime_indices) if prime_indices else 0

# Example usage:
# sol = Solution()
# print(sol.maximumPrimeDifference([4,2,9,5,3])) # Output: 3
# print(sol.maximumPrimeDifference([4,8,2,8]))   # Output: 0