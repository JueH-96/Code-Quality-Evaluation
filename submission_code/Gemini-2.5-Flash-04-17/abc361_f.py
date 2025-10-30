import sys

def safe_power(base, exp, limit):
    """Computes base^exp safely, checking against limit."""
    res = 1
    for _ in range(exp):
        # base^exp can exceed standard integer types, check against limit *before* multiplication
        # 0^exp = 0 for exp >= 1. Here exp >= 2.
        if base == 0:
             res = 0
             break
        # Check if res * base > limit. This is equivalent to limit // base < res for base > 0.
        if limit // base < res:
            return limit + 1 # Indicate that the power exceeds the limit
        res *= base
    return res

def floor_root(N, m):
    """Computes floor(N^(1/m)) using binary search."""
    if m < 1: return 0 # Invalid exponent
    if m == 1: return N # x = a^1 = a. The condition is b >= 2, so m=1 is not directly used in the sum.
    if N < 0: return 0 # N is guaranteed >= 1
    if N == 0: return 0
    if N == 1: return 1 # 1^m = 1

    # We are searching for the largest integer 'a' such that a^m <= N.
    # The search space for 'a' is [1, N].
    # A tighter upper bound is floor(N^(1/m)).
    # For m=2, N=10^18, this is 10^9.
    # For m >= 2, N^(1/m) <= N^(1/2). So max possible root is around 10^9.
    # A safe upper bound for binary search is 10^9 + a small margin.
    # Let's use 10^9 + 5. If N is small, this bound is too large but still correct.
    high = 10**9 + 5
    low = 1
    ans = 1 # Initialize with the smallest possible valid base

    while low <= high:
        mid = (low + high) // 2
        # mid is the candidate base 'a'
        
        # Calculate mid^m safely and compare with N
        power_val = safe_power(mid, m, N)

        if power_val <= N: # mid^m <= N, mid is a possible base. Try larger ones.
            ans = mid # mid is the largest root found so far that satisfies the condition
            low = mid + 1
        else: # mid^m > N, mid is too large. Try smaller ones.
            high = mid - 1

    return ans # This is floor(N^(1/m))


def get_mobius_sieve(max_m):
    """Precomputes Mobius function values up to max_m using a sieve."""
    # mu[n] = 0 if n has a square factor
    # mu[n] = (-1)^k if n is a product of k distinct primes
    mu = [0] * (max_m + 1)
    min_prime_factor = [0] * (max_m + 1)
    primes = []
    mu[1] = 1

    for i in range(2, max_m + 1):
        if min_prime_factor[i] == 0: # i is prime
            primes.append(i)
            min_prime_factor[i] = i
            mu[i] = -1 # mu(prime) = -1

        for p in primes:
            # If p > min_prime_factor[i], then i*p has a smaller prime factor min_prime_factor[i],
            # so we would have processed i*p when we were at min_prime_factor[i]. Break to avoid redundant work.
            # Also break if i*p exceeds the max_m limit.
            if p > min_prime_factor[i] or i * p > max_m:
                break
            
            min_prime_factor[i * p] = p # p is the smallest prime factor of i*p

            if p == min_prime_factor[i]:
                # i*p is divisible by p^2. mu(i*p) = 0.
                mu[i * p] = 0
            else:
                # p is a prime factor of i*p, and p is not the smallest prime factor of i.
                # mu(i*p) = -mu(i).
                mu[i * p] = -mu[i]
    return mu

def get_min_prime_factor_sieve(max_m):
    """Precomputes the minimum prime factor for numbers up to max_m."""
    min_prime_factor = [0] * (max_m + 1)
    primes = []

    for i in range(2, max_m + 1):
        if min_prime_factor[i] == 0: # i is prime
            primes.append(i)
            min_prime_factor[i] = i

        for p in primes:
            if p > min_prime_factor[i] or i * p > max_m:
                break
            min_prime_factor[i * p] = p
    return min_prime_factor

def is_square_free_using_sieve(n, min_prime_factor):
    """Checks if a number is square-free using min_prime_factor list."""
    if n < 1: return False
    if n == 1: return True
    
    # A number n > 1 is square-free if its prime factorization has all exponents equal to 1.
    # Using the minimum prime factor: Let p = min_prime_factor[n].
    # If p divides (n / p), then n is divisible by p^2, so it is not square-free.
    # min_prime_factor[n / p] gives the smallest prime factor of n / p.
    # If min_prime_factor[n / p] == p, it means p divides n / p.
    # So, n is square-free if and only if min_prime_factor[n / min_prime_factor[n]] != min_prime_factor[n].

    p = min_prime_factor[n]
    # For n in range [2, max_m_to_check], p = min_prime_factor[n] will be >= 2.
    
    if min_prime_factor[n // p] == p:
        return False
    return True


def solve():
    N = int(sys.stdin.readline())

    # The set of integers is { x | 1 <= x <= N, x = a^b for a >= 1, b >= 2 }.
    # This set can be decomposed into {1} U {a^b | a >= 2, b >= 2, a^b <= N}.
    # The number 1 is included because 1 = 1^2 (a=1 >= 1, b=2 >= 2) and 1 <= N.
    # Since {a^b | a >= 2, b >= 2} contains only values >= 2^2=4, the union is disjoint.
    # The size is 1 + |{a^b | a >= 2, b >= 2, a^b <= N}|.
    # The set {a^b | a >= 2, b >= 2, a^b <= N} is the set of perfect powers greater than 1 and less than or equal to N.
    # A number x > 1 is a perfect power iff x = c^p for some c >= 2 and prime p >= 2.
    # The set of perfect powers > 1 and <= N is the union of {c^p | c >= 2, c^p <= N} over all primes p >= 2.
    # Let S'_p = {c^p | c >= 2, c^p <= N}. We want |Union_{p prime, p>=2} S'_p|.
    # By inclusion-exclusion, |Union S'_p| = sum_{m square-free, m>=2} -mu(m) * |Intersection S'_p_i for p_i factors of m|.
    # Intersection S'_{p_1} cap ... cap S'_{p_k} = {c^{p_1...p_k} | c >= 2, c^{p_1...p_k} <= N}.
    # Let m = p_1...p_k. This is a square-free number.
    # |{c^m | c >= 2, c^m <= N}| = floor(N^(1/m)) - 1, provided floor(N^(1/m)) >= 2.
    # floor(N^(1/m)) >= 2 iff N^(1/m) >= 2 iff N >= 2^m iff m <= log2(N).

    # Maximum relevant exponent m is floor(log2(N)). For N=10^18, log2(10^18) approx 59.79.
    # So we need to check m up to 59.
    max_m_to_check = 59

    # Precompute Mobius function and min_prime_factor up to max_m_to_check.
    mu = get_mobius_sieve(max_m_to_check)
    min_prime_factor = get_min_prime_factor_sieve(max_m_to_check)

    count_perfect_powers_greater_than_1 = 0

    # Iterate through possible exponents m from 2 up to floor(log2(N)).
    # We only need to consider square-free m >= 2.
    for m in range(2, max_m_to_check + 1):
        # Check if m is square-free using the sieve result.
        # m is square-free iff mu[m] is not 0.
        # An alternative check using min_prime_factor:
        if is_square_free_using_sieve(m, min_prime_factor):
            # Calculate floor(N^(1/m)) using binary search
            a_max = floor_root(N, m)

            # Number of bases c >= 2 such that c^m <= N is a_max - 1.
            # Since m <= max_m_to_check = 59 and N >= 10^18 >= 2^59, N >= 2^m for m in this range.
            # This implies floor(N^(1/m)) >= floor((2^m)^(1/m)) = 2.
            # So a_max >= 2 is guaranteed for m in [2, max_m_to_check].
            num_bases_ge_2 = a_max - 1

            # The term in inclusion-exclusion is -mu(m) * (num_bases_ge_2)
            count_perfect_powers_greater_than_1 += -mu[m] * num_bases_ge_2

    # Total count = 1 (for x=1, which is 1^b for b>=2 and 1<=N) + count of perfect powers > 1
    total_count = 1 + count_perfect_powers_greater_than_1

    print(total_count)

# Helper function for the second sieve (min_prime_factor)
# This is needed by is_square_free_using_sieve. It should be defined before solve().
def get_min_prime_factor_sieve(max_m):
    """Precomputes the minimum prime factor for numbers up to max_m."""
    min_prime_factor = [0] * (max_m + 1)
    primes = []

    for i in range(2, max_m + 1):
        if min_prime_factor[i] == 0: # i is prime
            primes.append(i)
            min_prime_factor[i] = i

        for p in primes:
            if p > min_prime_factor[i] or i * p > max_m:
                break
            min_prime_factor[i * p] = p
    return min_prime_factor

# Helper function to check square-free using the sieve result.
# Placed here as it is used by solve().
def is_square_free_using_sieve(n, min_prime_factor):
    """Checks if a number is square-free using min_prime_factor list."""
    if n < 1: return False
    if n == 1: return True
    
    # A number n > 1 is square-free if its prime factorization has all exponents equal to 1.
    # This is equivalent to saying that n is not divisible by the square of any prime.
    # Using the minimum prime factor: Let p = min_prime_factor[n].
    # If p divides (n / p), then n is divisible by p^2, so it is not square-free.
    # min_prime_factor[n / p] gives the smallest prime factor of n / p.
    # If min_prime_factor[n / p] == p, it means p divides n / p.
    # So, n is square-free if and only if min_prime_factor[n / min_prime_factor[n]] != min_prime_factor[n].

    p = min_prime_factor[n]
    # For n in range [2, max_m_to_check], p = min_prime_factor[n] will be >= 2.
    
    if min_prime_factor[n // p] == p:
        return False
    return True


solve()