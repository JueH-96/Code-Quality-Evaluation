class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        primes = [i for i in nums if is_prime(i)]
        primes.sort()
        return max(primes[i+1]-primes[i] for i in range(len(primes)-1))