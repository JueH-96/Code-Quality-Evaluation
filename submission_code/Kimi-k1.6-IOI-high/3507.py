class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        max_p = int(r ** 0.5)
        sieve = []
        primes = []
        
        if max_p >= 2:
            sieve = [True] * (max_p + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(max_p ** 0.5) + 1):
                if sieve[i]:
                    sieve[i*i : max_p+1 : i] = [False] * len(sieve[i*i : max_p+1 : i])
            primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        
        count_special = 0
        for p in primes:
            square = p * p
            if l <= square <= r:
                count_special += 1
        
        total_numbers = r - l + 1
        return total_numbers - count_special