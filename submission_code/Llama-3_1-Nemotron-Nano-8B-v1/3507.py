import math

class Solution:
    def __init__(self):
        max_prime = int(math.sqrt(10**9))  # Maximum possible value for sqrt(r)
        sieve = [True] * (max_prime + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(max_prime)) + 1):
            if sieve[i]:
                sieve[i*i : max_prime+1 : i] = [False] * len(sieve[i*i : max_prime+1 : i])
        self.primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    def nonSpecialCount(self, l: int, r: int) -> int:
        a_min = math.ceil(math.sqrt(l))
        a_max = math.floor(math.sqrt(r))
        if a_min > a_max:
            return r - l + 1
        count = 0
        for p in self.primes:
            if a_min <= p <= a_max:
                count += 1
        return (r - l + 1) - count