class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        prime_indices = []
        for i, num in enumerate(nums):
            if is_prime(num):
                prime_indices.append(i)
        
        if len(prime_indices) < 2:
            return 0
        
        return max(prime_indices) - min(prime_indices)