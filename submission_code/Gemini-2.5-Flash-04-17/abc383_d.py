import sys
import bisect

# Max N is 4e12. sqrt(N) is 2e6.
# Numbers with 9 divisors are p^8 or p1^2 * p2^2.
# For p^8 <= N, p <= N^(1/8). For N=4e12, N^(1/8) approx 37.6. Primes are small.
# For p1^2 * p2^2 <= N with p1 < p2, p1 * p2 <= sqrt(N).
# Max sqrt(N) = 2e6. Smallest p1 is 2. Max p2 <= sqrt(N) / p1 <= 2e6 / 2 = 1e6.
# We need primes up to 1e6.
MAX_PRIME_SIEVE_LIMIT = 10**6

# Sieve of Eratosthenes to find primes up to MAX_PRIME_SIEVE_LIMIT
is_prime = [True] * (MAX_PRIME_SIEVE_LIMIT + 1)
is_prime[0] = is_prime[1] = False
for p in range(2, int(MAX_PRIME_SIEVE_LIMIT**0.5) + 1):
    if is_prime[p]:
        # Mark multiples of p starting from p*p
        for i in range(p*p, MAX_PRIME_SIEVE_LIMIT + 1, p):
            is_prime[i] = False

# Collect the primes into a list
primes = [p for p in range(MAX_PRIME_SIEVE_LIMIT + 1) if is_prime[p]]

def solve():
    # Read input N
    N = int(sys.stdin.readline())

    count = 0

    # Case 1: Numbers of the form p^8 where p is prime.
    # We need p^8 <= N.
    # Iterate through the list of precomputed primes.
    for p in primes:
        # Calculate p^8. Use Python's arbitrary precision int.
        p_pow_8 = p ** 8
        
        # Check if p^8 is within the limit N.
        if p_pow_8 <= N:
            # If yes, we found a number p^8 with 9 divisors <= N.
            count += 1
        else:
            # Since primes are sorted, for any larger prime q > p, q^8 will be > p^8 > N.
            # So, we can break the loop.
            break

    # Case 2: Numbers of the form p1^2 * p2^2 where p1 < p2 are distinct primes.
    # We need p1^2 * p2^2 <= N, which is equivalent to (p1 * p2)^2 <= N,
    # or p1 * p2 <= sqrt(N).
    
    # Calculate floor(sqrt(N)). Use Python's float for N**0.5 and convert to int.
    # Python's float precision is sufficient for N up to 4e12.
    limit_sqrt_N = int(N**0.5)

    # Iterate through primes p1 from the precomputed list.
    # The condition p1 < p2 and p1 * p2 <= limit_sqrt_N puts constraints on p1.
    # Since p2 > p1, p1 * p1 < p1 * p2 <= limit_sqrt_N.
    # So p1^2 < limit_sqrt_N, which means p1 < sqrt(limit_sqrt_N) = N^(1/4).
    # We can iterate through primes p1 and stop when p1^2 exceeds limit_sqrt_N,
    # as for any larger p1, p1^2 will be even larger, and p1 * p2 > p1^2 will also exceed limit_sqrt_N.
    
    # Iterate through primes using index i, as we need to count primes p2 > p1 (i.e., primes at index > i).
    for i in range(len(primes)):
        p1 = primes[i]

        # Calculate p1^2.
        p1_squared = p1 * p1
        
        # Check the stopping condition for p1. If p1^2 > limit_sqrt_N, break the loop.
        # Use Python's int for comparison.
        if p1_squared > limit_sqrt_N:
             break # Stop iterating p1

        # For the current prime p1, we need to find distinct primes p2 such that
        # p1 < p2 and p2 <= limit_sqrt_N / p1.
        
        # Calculate the upper bound for p2. Integer division handles floor.
        upper_p2 = limit_sqrt_N // p1

        # We need to count primes p2 in the `primes` list that satisfy two conditions:
        # 1. p2 > p1 (i.e., p2 must be in `primes[i+1:]`)
        # 2. p2 <= upper_p2
        
        # The primes satisfying condition 1 are primes starting from index i + 1.
        # The primes satisfying condition 2 are primes up to index `j` such that `primes[j] <= upper_p2`.
        # The count of primes <= upper_p2 in the `primes` list is given by bisect_right.
        # bisect_right(primes, upper_p2) returns the index where upper_p2 would be inserted
        # to the right, which is exactly the number of elements in `primes` less than or equal to upper_p2.
        idx_upper_p2_inclusive_plus_1 = bisect.bisect_right(primes, upper_p2)

        # We want primes whose index is > i and value <= upper_p2.
        # The indices of primes <= upper_p2 are 0, 1, ..., idx_upper_p2_inclusive_plus_1 - 1.
        # We need indices that are > i. So we need indices from i + 1 up to idx_upper_p2_inclusive_plus_1 - 1.
        # The number of such indices is (idx_upper_p2_inclusive_plus_1 - 1) - (i + 1) + 1
        # = idx_upper_p2_inclusive_plus_1 - (i + 1).
        
        # The index of the first prime strictly greater than p1 (primes[i]) is i + 1.
        idx_p1_exclusive = i + 1
        
        # The number of valid primes p2 is the number of elements in the `primes` list
        # whose index is >= idx_p1_exclusive and < idx_upper_p2_inclusive_plus_1.
        # This count is idx_upper_p2_inclusive_plus_1 - idx_p1_exclusive.
        num_p2 = idx_upper_p2_inclusive_plus_1 - idx_p1_exclusive
        
        # Add the number of valid p2 primes found for this p1 to the total count.
        # If upper_p2 is less than or equal to p1, idx_upper_p2_inclusive_plus_1 will be
        # less than or equal to i + 1, resulting in num_p2 <= 0, which is correct.
        count += num_p2

    # Print the final count
    print(count)

# Execute the solve function
solve()