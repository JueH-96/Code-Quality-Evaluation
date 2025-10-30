class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        primes = [i for i in range(2, n + 1) if is_prime(i)]
        pairs = []
        for i in range(len(primes)):
            for j in range(i, len(primes)):
                if primes[i] + primes[j] == n:
                    pairs.append([primes[i], primes[j]])
        pairs.sort(key=lambda x: x[0])
        return pairs