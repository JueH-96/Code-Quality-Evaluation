class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        prime_indices = []
        for idx, num in enumerate(nums):
            if is_prime(num):
                prime_indices.append(idx)
        
        if not prime_indices:
            return 0
        return prime_indices[-1] - prime_indices[0]