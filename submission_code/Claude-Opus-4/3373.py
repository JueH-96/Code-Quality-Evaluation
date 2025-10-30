class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Find indices of all prime numbers
        prime_indices = []
        for i in range(len(nums)):
            if is_prime(nums[i]):
                prime_indices.append(i)
        
        # If there's only one prime, distance is 0
        if len(prime_indices) == 1:
            return 0
        
        # Return the difference between the last and first prime indices
        return prime_indices[-1] - prime_indices[0]