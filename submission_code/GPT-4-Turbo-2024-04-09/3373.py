class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Helper function to determine if a number is prime
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
        
        # List to store indices of prime numbers
        prime_indices = []
        
        # Iterate over the array and find indices of prime numbers
        for i, num in enumerate(nums):
            if is_prime(num):
                prime_indices.append(i)
        
        # If there are no prime numbers or only one, the max distance is 0
        if len(prime_indices) < 2:
            return 0
        
        # Calculate the maximum distance between the first and last prime indices
        return prime_indices[-1] - prime_indices[0]