class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
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
        
        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]
        
        if len(prime_indices) < 2:
            return 0
        
        return prime_indices[-1] - prime_indices[0]