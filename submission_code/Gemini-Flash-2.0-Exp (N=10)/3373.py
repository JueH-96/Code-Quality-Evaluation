class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]
        
        if not prime_indices:
            return 0
        
        if len(prime_indices) == 1:
            return 0
        
        max_diff = 0
        for i in range(len(prime_indices)):
            for j in range(len(prime_indices)):
                max_diff = max(max_diff, abs(prime_indices[i] - prime_indices[j]))
        
        return max_diff