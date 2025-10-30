class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        import sys
        import threading
        def main():
            import math
            import random

            n = len(nums)
            nums = nums.copy()

            # Sieve to generate small primes up to sqrt(1e9)
            MAX_PRIME = 32000  # sqrt(1e9) ≈ 31623
            is_prime = [True] * (MAX_PRIME + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(MAX_PRIME ** 0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, MAX_PRIME +1, i):
                        is_prime[j] = False
            primes = [i for i, val in enumerate(is_prime) if val]

            # Factorization using Pollard's Rho algorithm
            from collections import defaultdict

            # Pollard's Rho algorithm for integer factorization
            def pollards_rho(n):
                if n % 2 == 0:
                    return 2
                if n % 3 == 0:
                    return 3
                if n % 5 == 0:
                    return 5
                if n % 7 == 0:
                    return 7
                if n % 11 == 0:
                    return 11
                if n % 13 == 0:
                    return 13
                if n % 17 == 0:
                    return 17
                if n % 19 == 0:
                    return 19
                if n % 23 == 0:
                    return 23
                if n % 29 == 0:
                    return 29
                if n % 31 == 0:
                    return 31
                if n % 37 == 0:
                    return 37
                if n % 41 == 0:
                    return 41
                if n % 43 == 0:
                    return 43
                if n % 47 == 0:
                    return 47
                if n % 53 == 0:
                    return 53
                if n % 59 == 0:
                    return 59
                if n % 61 == 0:
                    return 61

                if is_prime_number(n):
                    return n
                while True:
                    c = random.randrange(1, n)
                    f = lambda x: (pow(x, 2, n) + c) % n
                    x, y, d = 2, 2, 1
                    while d == 1:
                        x = f(x)
                        y = f(f(y))
                        d = math.gcd(abs(x - y), n)
                    if d != n:
                        return d
            # Check if a number is prime using Miller-Rabin primality test
            def is_prime_number(n):
                if n <= 1:
                    return False
                if n <= 3:
                    return True
                if n % 2 == 0:
                    return False
                r, s = 0, n-1
                while s % 2 == 0:
                    r +=1
                    s //=2
                for _ in range(5):
                    a = random.randrange(2, n - 1)
                    x = pow(a, s, n)
                    if x == 1 or x == n -1:
                        continue
                    for _ in range(r - 1):
                        x = pow(x, 2, n)
                        if x == n -1:
                            break
                    else:
                        return False
                return True

            exponents_list = []
            all_primes = set()
            for num in nums:
                num_copy = num
                exponents = defaultdict(int)
                for p in primes:
                    if p*p > num:
                        break
                    count = 0
                    while num % p == 0:
                        num //= p
                        count += 1
                    if count % 2 == 1:
                        exponents[p] = 1
                        all_primes.add(p)
                if num > 1:
                    if is_prime_number(num):
                        exponents[num] ^=1
                        all_primes.add(num)
                    else:
                        p = num
                        exponents[p] ^=1
                        all_primes.add(p)
                exponents_list.append(exponents)

            # Map primes to positions in bitmask
            prime_to_index = {p:i for i,p in enumerate(sorted(all_primes))}
            k = len(prime_to_index)  # Number of primes
            exponent_vectors = []
            for exponents in exponents_list:
                vec = 0
                for prime, exp in exponents.items():
                    if exp % 2 ==1:
                        vec |= 1 << prime_to_index[prime]
                exponent_vectors.append(vec)

            DP = {}
            DP[0] = 0  # Sum of nums[i] over subset having exponents sum to 0
            for i in range(n):
                e_i = exponent_vectors[i]
                weight = nums[i]
                new_DP = DP.copy()
                for sum_v, sum_w in DP.items():
                    new_sum = sum_v ^ e_i
                    new_w = sum_w + weight
                    if new_sum in new_DP:
                        if new_w > new_DP[new_sum]:
                            new_DP[new_sum] = new_w
                    else:
                        new_DP[new_sum] = new_w
                # Also consider the single element
                if e_i in new_DP:
                    if weight > new_DP[e_i]:
                        new_DP[e_i] = weight
                else:
                    new_DP[e_i] = weight
                DP = new_DP
            print(DP.get(0, max(nums)))

        threading.Thread(target=main).start()