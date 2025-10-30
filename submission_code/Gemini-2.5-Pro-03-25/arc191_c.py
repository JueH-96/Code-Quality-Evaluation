# YOUR CODE HERE
import sys
import random

# Miller-Rabin primality test
# Handles up to 64-bit integers deterministically with the chosen bases
def is_prime(n):
    """
    Checks if a number n is prime using Miller-Rabin test.
    Deterministic for n < 2^64 using the specified bases.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Basic trial division check using 6k +/- 1 optimization
    # This quickly handles smaller composite numbers
    i = 5
    # Check factors up to sqrt(n)
    limit = int(n**0.5) + 2 # Add 2 for safety with potential float inaccuracies
    while i < limit :
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    # If n is small enough that trial division has effectively checked it,
    # it must be prime. For larger n, proceed to Miller-Rabin.
    # The threshold where Miller-Rabin becomes necessary isn't strictly defined here,
    # but the trial division check helps efficiency.
    
    # Miller-Rabin test section begins
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Bases proven sufficient for deterministic check for n < 2^64
    # Reference: https://miller-rabin.appspot.com/ lists bases for different ranges.
    # This set is commonly cited for checks up to 2^64.
    _known_bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]

    for a in _known_bases:
        # The base 'a' must be less than 'n'
        if a >= n: break 
        
        # Compute x = a^d mod n using Python's efficient pow function
        x = pow(a, d, n) 
        
        # Check if x is 1 or n-1. If so, 'a' is not a witness for this initial check.
        if x == 1 or x == n - 1:
            continue # Try the next base
        
        # Witness loop: Check powers x, x^2, x^4, ..., x^(2^(s-1)) mod n
        is_witness = True # Assume 'a' is a witness unless proven otherwise
        for _ in range(s - 1): # Loop s-1 times
            x = pow(x, 2, n) # Compute next power: x = x^2 mod n
            if x == n - 1:
                # Found x = n-1. According to Miller-Rabin theory, 'a' is not a witness.
                is_witness = False 
                break # Stop checks for this base 'a'
        
        # If after all checks, 'a' remained a witness (is_witness is True)
        if is_witness: 
            return False # n is definitely composite

    # If n passed Miller-Rabin tests for all chosen bases, it is prime
    # (deterministically for n < 2^64 with these bases)
    return True 


# Function to get distinct prime factors of a number
def get_prime_factors(num):
    """
    Finds distinct prime factors of num using trial division.
    Optimized to check 2 separately and then only odd numbers.
    Efficient enough for N up to 10^9.
    """
    factors = set()
    d = 2
    temp_num = num
    # Handle factor 2 separately
    if temp_num % d == 0:
        factors.add(d)
        while temp_num % d == 0:
            temp_num //= d
    
    # Handle odd factors starting from 3
    d = 3
    # Only need to check up to sqrt(temp_num)
    limit = int(temp_num**0.5) + 1 
    while d <= limit:
        if temp_num % d == 0:
            factors.add(d)
            while temp_num % d == 0:
                temp_num //= d
            # Update limit since temp_num decreased, potentially speeding up
            limit = int(temp_num**0.5) + 1
        d += 2 # Check next odd number
    
    # If temp_num is still > 1 after trial divisions, 
    # the remaining temp_num must be a prime factor itself.
    if temp_num > 1:
        factors.add(temp_num)
    return list(factors)

# Main logic to solve a single test case
def solve():
    N = int(sys.stdin.readline())

    # Handle base case N=1 separately
    if N == 1:
        # The condition is: smallest positive integer n such that A^n - 1 is a multiple of M is 1.
        # This means A^1 - 1 must be a multiple of M.
        # If M=1, A^1-1 = A-1 is always a multiple of 1.
        # Smallest n is indeed 1. Example pair (2, 1) works.
        print(2, 1)
        return

    # Find distinct prime factors of N. This is needed for checking the order condition.
    prime_factors_N = get_prime_factors(N)

    k = 1
    while True:
        # Search for a prime q of the form k*N + 1.
        # By Dirichlet's theorem on arithmetic progressions, such a prime exists.
        q = k * N + 1

        # The chosen modulus will be M = 2q. Check if M exceeds the 10^18 limit.
        # This implies q must be <= 5 * 10^17.
        if q > 5 * 10**17:
             # If the smallest prime q=kN+1 leads to M > 10^18, this strategy path fails.
             # The problem guarantees a solution exists within the constraints [1, 10^18].
             # This implies we should find a suitable q before this limit is hit.
             # Add a safety break; this should not be reached given problem guarantees.
             break 
        
        # Check if the calculated q is prime using Miller-Rabin test.
        # Skip if q is not prime. Also skip if q=2 (only possible if N=1, already handled).
        if q > 2 and is_prime(q):
             # Found a suitable prime q. Set the modulus M = 2q.
             M = 2 * q
            
             # Now find an integer A such that its multiplicative order modulo M is exactly N.
             # This is equivalent to finding A such that ord_M(A) = N.
             # Since M=2q, ord_M(A) = lcm(ord_2(A), ord_q(A)).
             # We require A to be odd, so ord_2(A) = 1.
             # Thus, we need ord_M(A) = lcm(1, ord_q(A)) = ord_q(A).
             # The task reduces to finding an odd A such that ord_q(A) = N.
             
             # We search for a base A0 such that ord_q(A0) = N.
             # Start searching A0 from 2 upwards.
             A0 = 2
             # Upper bound for A0 search? Theoretically need to check up to q-1. 
             # But practically, a small A0 usually works. Use q as a loose upper check.
             while A0 < q: 
                 # Check condition 1: A0^N === 1 (mod q)
                 # If this fails, A0 cannot have order N (or any divisor of N).
                 if pow(A0, N, q) != 1:
                     A0 += 1
                     continue # Try the next A0

                 # Check condition 2: A0^(N/p) != 1 (mod q) for all distinct prime factors p of N
                 # This ensures that the order is exactly N, and not a proper divisor of N.
                 valid_order = True
                 for p_factor in prime_factors_N:
                     # Compute A0^(N/p) mod q using modular exponentiation
                     if pow(A0, N // p_factor, q) == 1:
                         # If A0^(N/p) == 1 mod q for any prime factor p of N,
                         # then the order of A0 divides N/p, which is smaller than N.
                         valid_order = False
                         break # Stop checking factors for this A0, move to the next A0
                 
                 if valid_order:
                     # Found an A0 such that ord_q(A0) = N.
                     # Now determine the final A based on A0's parity. A must be odd.
                     A = A0
                     # q=kN+1. Since N>=1, q>=2. If q=2, N=1 (handled). If N>1, q>=3.
                     # So q is an odd prime if N>1.
                     if A0 % 2 == 0: 
                         # If A0 is even, set A = A0 + q. A is then odd (even + odd = odd).
                         # A === A0 (mod q), so ord_q(A) = ord_q(A0) = N.
                         A = A0 + q
                     
                     # Final constraint check: A must be <= 10^18
                     if A > 10**18:
                          # If this calculated A exceeds the limit, this A0 choice is invalid.
                          # Continue searching for the next suitable A0.
                          # This case implies A0 was large and even, or q was near the limit 5e17.
                          A0 += 1
                          continue 

                     # Found a valid pair (A, M) satisfying all conditions. Print and return.
                     print(A, M)
                     return 

                 # If A0 did not have the correct order N, increment and try the next A0
                 A0 += 1
                 
             # If the inner while loop finishes without finding A0 (A0 reached q),
             # something is wrong, possibly with assumptions or the problem guarantee.
             # This break is a safeguard. Might need to try the next k value.
             break

        # If q was not prime, or if inner A0 search failed (safeguard break), increment k
        k += 1
        # Safety break for the outer k loop to prevent potential infinite loops
        if k > 10000: # Set an arbitrary large limit for k checks
             # print(f"Error: k exceeded limit for N={N}", file=sys.stderr) # Optional debug msg
             break # Stop searching for k


# Read the number of test cases from standard input
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()