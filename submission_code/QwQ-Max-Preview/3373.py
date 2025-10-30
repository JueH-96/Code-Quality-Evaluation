from typing import List

class Solution:
    _primes = None

    @classmethod
    def _init_primes(cls):
        if cls._primes is not None:
            return
        max_num = 100
        sieve = [True] * (max_num + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_num ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i : max_num + 1 : i] = [False] * len(sieve[i*i : max_num + 1 : i])
        cls._primes = sieve

    def maximumPrimeDifference(self, nums: List[int]) -> int:
        Solution._init_primes()
        first = -1
        last = -1
        for idx, num in enumerate(nums):
            if Solution._primes[num]:
                if first == -1:
                    first = idx
                last = idx
        return last - first