class Solution:
    def is_prime(self, n):
        if n <= 1:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = []
        for idx, num in enumerate(nums):
            if self.is_prime(num):
                primes.append(idx)
        if len(primes) == 1:
            return 0
        else:
            return primes[-1] - primes[0]