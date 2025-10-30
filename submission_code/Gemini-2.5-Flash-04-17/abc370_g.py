import sys
import math
from collections import defaultdict

# Set recursion depth limit
# The maximum recursion depth comes from the prime index (up to pi(sqrt(N))) and the exponent k (up to log_p(N)).
# log_2(10^10) is about 34. pi(10^5) is about 9592.
# Total recursion depth could be sum of max k over distinct prime factors times the number of prime factors?
# Or maybe the depth is bounded by sum of log_p(N) over distinct primes?
# A rough upper bound on chain of divisions N -> N/p1 -> N/(p1*p2) -> ... is log N.
# The state space should be the main concern, not recursion depth.
# sys.setrecursionlimit(2000) # Default is usually higher, but can increase if needed.

N, M = map(int, sys.stdin.readline().split())

MOD = 998244353
MAX_EXPONENT = int(math.log2(N)) + 2 # Max exponent for any prime factor
MAX_COMB_N = M + MAX_EXPONENT # Max value for n in combinations(n, k)
fact = [1] * MAX_COMB_N
invFact = [1] * MAX_COMB_N

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def modInverse(n):
    return power(n, MOD - 2)

def precompute_factorials(n):
    for i in range(2, n):
        fact[i] = (fact[i - 1] * i) % MOD
    invFact[n - 1] = modInverse(fact[n - 1])
    for i in range(n - 2, -1, -1):
        invFact[i] = (invFact[i + 1] * (i + 1)) % MOD

def combinations(n, k):
    if k < 0 or k > n:
        return 0
    # n = M + k_prime_power - 1. M + k_prime_power - 1 can be up to MAX_COMB_N - 1.
    # k is the exponent k_prime_power. k is at most MAX_EXPONENT.
    return (((fact[n] * invFact[k]) % MOD) * invFact[n - k]) % MOD

# Precompute factorials up to MAX_COMB_N - 1
precompute_factorials(MAX_COMB_N)

# Multiplicative summatory function using O(sqrt(N) * pi(sqrt(N))) approach
# S(x, pi_idx) = sum_{n <= x, lpf(n) >= primes[pi_idx]} f(n)
sum_memo = {} # Use a dictionary for memoization

def get_sum_recursive(x, pi_idx, primes, f_pk_func):
    if x == 0: return 0
    # Base case: no more primes to consider or current prime > x
    # The sum is over n <= x with lpf(n) >= primes[pi_idx].
    # If primes[pi_idx] > x, the only possible such n is 1.
    # The function f is multiplicative, so f(1)=1.
    if pi_idx >= len(primes) or primes[pi_idx] > x:
         return 1 # f(1)

    # Memoization key: (x, pi_idx)
    key = (x, pi_idx)
    if key in sum_memo:
        return sum_memo[key]

    p = primes[pi_idx]
    # Sum over n with lpf(n) > p (i.e., lpf(n) >= next prime)
    res = get_sum_recursive(x, pi_idx + 1, primes, f_pk_func)

    # Sum over n with lpf(n) = p
    # n = p^k * m, m <= x/p^k, lpf(m) > p
    pk = p
    k = 1
    while pk <= x:
        term_f_pk = f_pk_func(p, k)
        if term_f_pk != 0:
             # The recursive call should sum over m <= x/p^k with lpf(m) >= primes[pi_idx + 1]
             res = (res + term_f_pk * get_sum_recursive(x // pk, pi_idx + 1, primes, f_pk_func)) % MOD
        k += 1
        # Check for overflow before multiplication
        if pk > x // p : break
        pk *= p

    sum_memo[key] = res
    return res

# Helper function for h(p^k)
# This determines if the exponent k for prime p is allowed for a non-good number
def is_not_good_exponent(p, k):
    if k == 0: return True # Any prime^0 = 1, which is not good

    if p == 2:
        return k % 2 == 0
    elif p == 3:
        return True # any exponent is fine for prime 3
    elif p % 3 == 1: # p >= 5
        return k % 3 != 2
    elif p % 3 == 2: # p >= 5
        return k % 2 == 0
    # Should not happen for prime p
    return False

# Function to compute sum(f(n)) for n <= N where f is multiplicative
def compute_sum(N, f_pk_raw_func):
    # Precompute primes up to sqrt(N)
    P = int(N**0.5)
    primes = []
    is_prime = [True] * (P + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, P + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, P + 1, i):
                is_prime[j] = False

    # Wrap the raw f_pk_func to handle the f(1) case (p=1, k=0)
    def f_pk_func(p, k):
        if p == 1 and k == 0: return 1
        return f_pk_raw_func(p, k)

    global sum_memo
    sum_memo.clear() # Clear memoization cache for each function call
    return get_sum_recursive(N, 0, primes, f_pk_func)

# Define the f_pk functions for tau_M and h
def f_tau_M_pk(p, k):
    # tau_M(p^k) = combinations(M + k - 1, k)
    return combinations(M + k - 1, k)

def f_h_pk(p, k):
    # h(p^k) = [exponent k is allowed for prime p] * tau_M(p^k)
    if not is_not_good_exponent(p, k):
        return 0
    return combinations(M + k - 1, k)

# Calculate the total number of sequences with product <= N
total_sum = compute_sum(N, f_tau_M_pk)

# Calculate the number of sequences with non-good product <= N
not_good_sum = compute_sum(N, f_h_pk)

# The answer is Total sum - Non-good sum
answer = (total_sum - not_good_sum + MOD) % MOD

print(answer)