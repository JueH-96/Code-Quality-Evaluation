import sys
import bisect
import math

# For A <= 10^12, sqrt(A) <= 10^6. standard float precision is sufficient for sqrt up to ~10^15.
# int(A**0.5) is the standard way to get floor(sqrt(A)) in competitive programming for these bounds.
# If math.isqrt is available (Python 3.8+), it's the most accurate integer square root.
# Let's use math.isqrt if possible, otherwise fall back to int(A**0.5).
def integer_sqrt(n):
    if hasattr(math, 'isqrt'):
        return math.isqrt(n)
    else:
        # Fallback for older Python versions.
        # For A <= 10^12, S <= 10^6, int(n**0.5) should be sufficient.
        # A small adjustment might be needed for perfect squares near boundary,
        # but for integer n, int(n**0.5) usually gives floor(sqrt(n)).
        # A safer way might be to check (res+1)^2 > n, but this is likely unnecessary here.
        return int(n**0.5)

# Pre-computation
# MAX_S is the maximum possible value of floor(sqrt(A))
# For A <= 10^12, MAX_S = floor(sqrt(10^12)) = 10^6
MAX_S = 1000000

# Sieve of Eratosthenes to find primes up to MAX_S
# We need primes up to MAX_S for the bases p and q.
IS_PRIME = [True] * (MAX_S + 1)
IS_PRIME[0] = IS_PRIME[1] = False
PRIMES = []
# The sieve marking needs to go up to sqrt(MAX_S) for primes i
sieve_mark_limit = int(math.sqrt(MAX_S))
for i in range(2, MAX_S + 1):
    if IS_PRIME[i]:
        PRIMES.append(i)
        # Optimization: start marking multiples from i*i
        # Only need to do this if i <= sqrt(MAX_S)
        if i <= sieve_mark_limit:
            # Check i * i <= MAX_S to prevent potential overflow if MAX_S were larger
            # although for MAX_S=10^6, i*i <= 10^6 fits in int.
            for j in range(i * i, MAX_S + 1, i):
                IS_PRIME[j] = False

# Set to store M values (numbers of the form p^a q^b with p < q, a>=1, b>=1)
M_values = set()

# Iterate through primes p
for i_p in range(len(PRIMES)):
    p = PRIMES[i_p]
    
    # Iterate through exponent a >= 1 for p
    # Calculate X = p^a iteratively, check for overflow against MAX_S
    # Start with a=1, X=p
    X = p
    
    # The loop continues as long as X = p^a is a potential first factor <= MAX_S.
    # Check for overflow before multiplication X * p.
    while X <= MAX_S:
        # Iterate through primes q > p
        # We only need to iterate through primes q that are potentially <= MAX_S / X
        # The primes list is sorted, so we can iterate from the index after p.
        for i_q in range(i_p + 1, len(PRIMES)):
            q = PRIMES[i_q]
            
            # The second factor Y = q^b must satisfy Y <= MAX_S / X.
            # The smallest possible Y is q (when b=1).
            # If q itself is greater than MAX_S / X, then q^b for b>=1 will also be > MAX_S / X.
            limit_qb = MAX_S // X
            if q > limit_qb:
                 break # Stop iterating q for the current X

            # Iterate through exponent b >= 1 for q
            # Calculate Y = q^b iteratively, check for overflow against limit_qb
            # Start with b=1, Y=q
            Y = q
            
            # The loop continues as long as Y = q^b is a potential second factor <= limit_qb.
            # Check for overflow before multiplication Y * q.
            while Y <= limit_qb:
                # M = X * Y = p^a * q^b. Both X and Y are checked against limits, so M <= MAX_S.
                M_values.add(X * Y)
                
                # Check potential overflow for Y * q before multiplication
                # If Y > limit_qb / q, then Y * q > limit_qb
                # This prevents Y from exceeding limit_qb in the next iteration.
                if limit_qb // q < Y:
                     break
                Y *= q # Increment Y to q^(b+1)
            
        # Check potential overflow for X * p before next iteration of a
        # If X > MAX_S / p, then X * p > MAX_S, so the next X will exceed MAX_S.
        # This check allows the outer while loop condition X <= MAX_S to be checked correctly.
        if MAX_S // p < X:
             break
        X *= p # Increment X to p^(a+1)

# Convert set to sorted list for binary search
sorted_M_values = sorted(list(M_values))

# Process queries
# Read input using sys.stdin.readline for performance
input = sys.stdin.readline
Q = int(input())

for _ in range(Q):
    A = int(input())
    S = integer_sqrt(A) # S = floor(sqrt(A))

    # Find the largest M in sorted_M_values such that M <= S
    # bisect_right returns index k such that all elements sorted_M_values[:k] <= S.
    # The largest element <= S is at index k-1.
    # bisect_right(a, x) returns len(a) if all elements are <= x.
    k = bisect.bisect_right(sorted_M_values, S)
    
    # The problem guarantees that a 400 number not exceeding A always exists.
    # The smallest 400 number is 36, so A >= 36.
    # This means S = floor(sqrt(A)) >= floor(sqrt(36)) = 6.
    # The smallest M value in M_values is 6 (from 2^1 * 3^1).
    # Since 6 <= S and 6 is in the list, bisect_right will return an index k >= 1.
    # So k-1 is a valid index referring to an element <= S.
    M = sorted_M_values[k-1]
    
    # The answer is M^2. M can be up to 10^6, so M^2 can be up to 10^12.
    # Python's int handles large numbers automatically.
    result = M * M
    print(result)