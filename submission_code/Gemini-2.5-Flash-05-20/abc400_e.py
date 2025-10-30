import math
import sys
import bisect

# Set the maximum possible value for X.
# A 400 number N = X^2. Given N <= 10^12, X must be <= sqrt(10^12) = 10^6.
MAX_X = 10**6

# --- Precomputation Phase ---

# 1. Sieve of Eratosthenes to find all primes up to MAX_X.
# This creates a boolean list `is_prime` where is_prime[i] is True if i is prime.
# It also populates a list `primes` with all prime numbers up to MAX_X.
is_prime = [True] * (MAX_X + 1)
is_prime[0] = is_prime[1] = False # 0 and 1 are not prime
primes = []
for p in range(2, MAX_X + 1):
    if is_prime[p]:
        primes.append(p)
        # Mark multiples of p as not prime. Start from p*p for optimization.
        for multiple in range(p * p, MAX_X + 1, p):
            is_prime[multiple] = False

# 2. Generate all valid X values.
# A valid X is of the form p1^a * p2^b, where p1, p2 are distinct primes and a, b >= 1.
# Also, X must be <= MAX_X.
valid_X_values = set() # Use a set to automatically handle duplicates and for efficient addition.

# Iterate through all possible first prime factors, p1.
for i, p1 in enumerate(primes):
    # val1 will represent p1^a (where a >= 1)
    val1 = p1 
    # Loop to generate powers of p1 (val1 = p1^a)
    while val1 <= MAX_X:
        # Iterate through all possible second prime factors, p2.
        # To ensure p1 and p2 are distinct and avoid generating duplicates (e.g., 2*3 and 3*2),
        # we enforce p1 < p2. So, p2 loop starts from the prime after p1.
        for j in range(i + 1, len(primes)):
            p2 = primes[j]
            
            # Optimization: If val1 * p2 would exceed MAX_X even for p2^1,
            # then all subsequent p2's will also result in a product > MAX_X.
            # Use integer division to prevent potential conceptual overflow before multiplication,
            # though Python integers handle arbitrary size, this ensures we stay within MAX_X.
            if val1 > MAX_X // p2:
                break # All subsequent p2s will lead to val1 * p2 > MAX_X

            val2 = p2 # Start with p2^1
            # Loop to generate powers of p2 (val2 = p2^b)
            while True:
                current_X = val1 * val2
                if current_X > MAX_X:
                    break # current_X exceeded MAX_X, stop generating further powers of p2
                
                valid_X_values.add(current_X) # Add the valid X number
                
                # Check for conceptual overflow before multiplying val2 by p2.
                # If val2 * p2 would exceed MAX_X, then further val2 powers are too large.
                if val2 > MAX_X // p2:
                    break
                val2 *= p2 # Move to the next power of p2 (p2^(b+1))
        
        # Check for conceptual overflow before multiplying val1 by p1.
        # If val1 * p1 would exceed MAX_X, then further val1 powers are too large.
        if val1 > MAX_X // p1:
            break
        val1 *= p1 # Move to the next power of p1 (p1^(a+1))

# Convert the set of valid X values to a sorted list.
# This allows for efficient binary search during query processing.
sorted_valid_X_values = sorted(list(valid_X_values))

# --- Query Processing Phase ---

# Read the number of queries.
Q = int(sys.stdin.readline())

results = [] # List to store results for each query.
for _ in range(Q):
    A = int(sys.stdin.readline())
    
    # We need to find the largest 400 number N <= A.
    # Since N = X^2, we are looking for the largest X such that X^2 <= A.
    # This is equivalent to finding the largest X such that X <= sqrt(A).
    # math.isqrt(A) computes floor(sqrt(A)), which is exactly what we need for the upper bound of X.
    max_x_for_A = math.isqrt(A)
    
    # Use binary search (bisect_right) on the sorted list of valid X values.
    # bisect_right(a, x) returns an insertion point which comes after (to the right of)
    # any existing entries of x in a. All elements to the left of this point are <= x.
    # So, idx will be the index of the first element strictly greater than max_x_for_A.
    # Therefore, the element at `idx - 1` is the largest element in `sorted_valid_X_values`
    # that is less than or equal to `max_x_for_A`.
    idx = bisect.bisect_right(sorted_valid_X_values, max_x_for_A)
    
    # The problem constraints guarantee that a 400 number not exceeding A always exists.
    # The smallest A is 36, for which max_x_for_A is 6.
    # The smallest valid X is 6 (for N=36). So, idx will always be at least 1,
    # meaning `sorted_valid_X_values[idx - 1]` is always a valid index.
    ans_X = sorted_valid_X_values[idx - 1]
    
    # The answer is the square of this X.
    results.append(str(ans_X * ans_X))

# Print all results, each on a new line.
sys.stdout.write("
".join(results) + "
")