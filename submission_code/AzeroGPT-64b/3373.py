class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n <= 1: return False
            if n <= 3: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        max_diff = 0
        prime_indices = []
        
        for i, num in enumerate(nums):
            if is_prime(num):
                prime_indices.append(i)
        
        if len(prime_indices) > 1:
            max_diff = max(prime_indices) - min(prime_indices)
        
        return max_diff