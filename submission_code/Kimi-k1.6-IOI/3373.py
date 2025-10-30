class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes_indices = []
        for idx, num in enumerate(nums):
            if is_prime(num):
                primes_indices.append(idx)
        
        return primes_indices[-1] - primes_indices[0]