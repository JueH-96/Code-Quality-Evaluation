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
        
        prime_indices = []
        
        # Find all indices with prime numbers
        for i in range(len(nums)):
            if is_prime(nums[i]):
                prime_indices.append(i)
        
        # Return the maximum distance between prime indices
        return max(prime_indices) - min(prime_indices)