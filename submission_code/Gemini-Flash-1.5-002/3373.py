class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n <= 1:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        primes = [i for i, num in enumerate(nums) if is_prime(num)]
        if not primes:
            return 0
        
        max_diff = 0
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                max_diff = max(max_diff, primes[j] - primes[i])
        return max_diff