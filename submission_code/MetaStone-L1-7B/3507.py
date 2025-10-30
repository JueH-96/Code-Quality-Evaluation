import math

class Solution:
    _sieve = None

    @staticmethod
    def compute_sieve():
        if Solution._sieve is None:
            max_limit = int(math.sqrt(1e9)) + 1  # 31622 + 1 = 31623
            Solution._sieve = [True] * (max_limit + 1)
            Solution._sieve[0] = Solution._sieve[1] = False
            for i in range(2, int(math.sqrt(max_limit)) + 1):
                if Solution._sieve[i]:
                    for j in range(i * i, max_limit + 1, i):
                        Solution._sieve[j] = False

    def nonSpecialCount(self, l: int, r: int) -> int:
        Solution.compute_sieve()
        a = math.ceil(math.sqrt(l))
        b = math.floor(math.sqrt(r))
        if a > b:
            count_primes = 0
        else:
            count_primes = 0
            for i in range(a, b + 1):
                if Solution._sieve[i]:
                    count_primes += 1
        total = r - l + 1
        return total - count_primes