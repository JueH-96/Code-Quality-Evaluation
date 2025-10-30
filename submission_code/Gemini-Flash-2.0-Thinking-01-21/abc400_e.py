import sys
import math
from bisect import bisect_right

# Limit for generating M values (sqrt(A_max))
LIMIT = 10**6

# Sieve of Eratosthenes to find primes up to LIMIT
# is_prime[i] will be True if i is prime, False otherwise
is_prime = [True] * (LIMIT + 1)
is_prime[0] = is_prime[1] = False # 0 and 1 are not prime
# Iterate up to the square root of LIMIT
for i in range(2, int(math.sqrt(LIMIT)) + 1):
    if is_prime[i]:
        # Mark multiples of i as not prime, starting from i*i
        for j in range(i * i, LIMIT + 1, i):
            is_prime[j] = False

# Collect the prime numbers into a list
primes = [i for i in range(LIMIT + 1) if is_prime[i]]

# Generate M values of the form p^a * q^b where p < q are distinct primes and a >= 1, b >= 1
m_values = set()

# Iterate through prime p using index i
for i in range(len(primes)):
    p = primes[i]
    
    # Calculate powers of p: p^a, starting with a = 1 (p_a = p^1)
    p_a = p
    
    # Loop for exponent 'a' (a >= 1)
    while True:
        # If the current power of p exceeds the LIMIT, no more multiples involving p^a can be <= LIMIT
        if p_a > LIMIT:
            break

        # Iterate through primes q > p using index j, starting from the prime after p (i + 1)
        for j in range(i + 1, len(primes)):
            q = primes[j]
            
            # Optimization: If p_a * q (which is p^a * q^1, the smallest possible M for this p_a and q)
            # is already greater than LIMIT, then p^a * q^b for any b >= 1 will also be > LIMIT.
            # Since q is increasing in the loop, all subsequent primes will also result in values > LIMIT.
            # Check using division to prevent potential overflow before multiplication and to check against LIMIT.
            # If p_a > LIMIT // q, then p_a * q > LIMIT. Break the inner q loop for this p_a.
            # This check is safe because q >= primes[i+1] >= 3 (if i=0, p=2, q>=3; if i>0, p>=3, q>p>=5).
            if p_a > LIMIT // q:
                break

            # Calculate powers of q: q^b, starting with b = 1 (q_b = q^1)
            q_b = q
            # Loop for exponent 'b' (b >= 1)
            while True:
                # Calculate M = p_a * q_b
                m_val = p_a * q_b
                
                # If M exceeds the LIMIT, break the inner b loop
                if m_val > LIMIT:
                    break
                
                # Add the valid M value to the set. Set automatically handles distinct values.
                m_values.add(m_val)
                
                # Prepare for the next power of q. Check if the next q_b (q_b * q)
                # multiplied by p_a would exceed the LIMIT.
                # The next m_val would be p_a * (q_b * q). We need p_a * q_b * q <= LIMIT.
                # This is equivalent to checking if m_val * q <= LIMIT.
                # This is equivalent to checking if m_val > LIMIT // q (if q > 0).
                # Since q is a prime >= 3, q > 0.
                if m_val > LIMIT // q:
                     break

                # Calculate the next power of q: q_b = q_b * q
                q_b *= q 

        # Prepare for the next power of p. Check if the next p_a (p_a * p)
        # would exceed the LIMIT. If the next p_a exceeds LIMIT, then p_a * q_b will definitely exceed LIMIT
        # for any q_b >= q >= primes[i+1] >= 3.
        # This is equivalent to checking if p_a * p <= LIMIT.
        # This is equivalent to checking if p_a > LIMIT // p (if p > 0).
        # Since p is a prime >= 2, p > 0.
        if p_a > LIMIT // p:
             break
             
        # Calculate the next power of p: p_a = p_a * p
        p_a *= p

# Convert the set of M values to a sorted list for efficient searching
sorted_m_values = sorted(list(m_values))

# Read the number of queries
Q = int(sys.stdin.readline())

# Process each query
# Using sys.stdout.write directly might be faster for competitive programming output
# Ensure that writes include a newline character '
'
# sys.stdout.reconfigure(line_buffering=True) # Uncomment this line if output needs flushing immediately without newline

for _ in range(Q):
    # Read A for the current query
    A = int(sys.stdin.readline())
    
    # A 400 number N is of the form M^2, where M = p^a * q^b.
    # We are looking for the largest N <= A, which is the largest M^2 <= A,
    # which is equivalent to finding the largest M <= sqrt(A).
    # We need the integer part of sqrt(A) as the limit for M.
    m_limit = int(math.sqrt(A))
    
    # Find the index in sorted_m_values where m_limit would be inserted to maintain sorted order.
    # bisect_right returns the index of the first element strictly greater than m_limit.
    # All elements before this index (at index idx - 1 and lower) are less than or equal to m_limit.
    idx = bisect_right(sorted_m_values, m_limit)
    
    # The largest M value in our sorted list that is less than or equal to m_limit is at index idx - 1.
    # The problem guarantees that A >= 36. The smallest 400 number is 36 = 6^2, where M=6 (2^1 * 3^1).
    # Our generated list `sorted_m_values` will always contain 6.
    # For A >= 36, m_limit = int(sqrt(A)) >= int(sqrt(36)) = 6.
    # When searching for m_limit >= 6 in `sorted_m_values` which contains 6, bisect_right will return
    # an index `idx` that is at least 1 (the position right after the first element 6, or later).
    # Therefore, accessing `sorted_m_values[idx - 1]` is always safe as idx >= 1.
    largest_m = sorted_m_values[idx - 1]
    
    # The answer to the query is the square of the largest valid M found
    result = largest_m * largest_m
    
    # Print the result followed by a newline character
    sys.stdout.write(str(result) + '
')