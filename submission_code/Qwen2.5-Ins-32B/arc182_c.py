import sys
from collections import defaultdict

def count_divisors(n, primes):
    divisors = 1
    for p in primes:
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        divisors *= (count + 1)
    if n > 1:
        divisors *= 2
    return divisors

def solve(n, m):
    MOD = 998244353
    primes = [2, 3, 5, 7, 11, 13]
    prime_powers = defaultdict(int)
    for i in range(1, m + 1):
        for p in primes:
            count = 0
            while i % p == 0:
                i //= p
                count += 1
            if count > 0:
                prime_powers[p] += count
    dp = [0] * (m + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        new_dp = [0] * (m + 1)
        for j in range(m + 1):
            for k in range(1, m + 1):
                new_dp[j ^ k] = (new_dp[j ^ k] + dp[j]) % MOD
        dp = new_dp
    result = 0
    for j in range(m + 1):
        product = 1
        for p in primes:
            product = (product * pow(p, (prime_powers[p] & j).bit_count(), MOD - 1)) % (MOD - 1)
        result = (result + dp[j] * pow(m, n, MOD) * pow(product, MOD - 2, MOD)) % MOD
    return result

input = sys.stdin.read
data = input().split()
n, m = map(int, data)
print(solve(n, m))