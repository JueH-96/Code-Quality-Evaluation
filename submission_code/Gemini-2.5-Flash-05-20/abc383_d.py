import math
import sys
from bisect import bisect_right

def solve():
    N = int(sys.stdin.readline())

    # --- Case 1: k = p^8 ---
    # We need p^8 <= N.
    # The maximum value of N is 4 * 10^12.
    # (31^8) = 1,013,149,791,851 (approx 1.01 * 10^12) <= 4 * 10^12
    # (37^8) = 4,809,876,798,001 (approx 4.81 * 10^12) > 4 * 10^12
    # So, primes up to 31 are relevant.
    small_primes_for_p8 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    count_case1 = 0
    for p in small_primes_for_p8:
        # Python handles arbitrarily large integers, so p**8 calculation is exact and won't overflow.
        if p**8 <= N:
            count_case1 += 1
        else:
            # Since primes are in increasing order, if p**8 > N,
            # all subsequent (larger) primes will also yield p**8 > N.
            break 
    
    # --- Case 2: k = p1^2 * p2^2 = (p1 * p2)^2 for distinct primes p1 < p2 ---
    # We need (p1 * p2)^2 <= N, which means p1 * p2 <= sqrt(N).
    # Let M = floor(sqrt(N)).
    # N <= 4 * 10^12, so M <= sqrt(4 * 10^12) = 2 * 10^6.
    
    # math.sqrt uses double-precision floats, which is sufficient for N up to 4*10^12
    # (2^53 is approx 9*10^15, which is greater than 4*10^12).
    M = int(math.sqrt(N)) 

    # Sieve of Eratosthenes to generate primes up to M
    is_prime = [True] * (M + 1)
    if M >= 0: 
        is_prime[0] = False
    if M >= 1: 
        is_prime[1] = False
    
    # The sieve loop runs up to sqrt(M)
    for i in range(2, int(math.sqrt(M)) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime, starting from i*i
            for multiple in range(i*i, M + 1, i):
                is_prime[multiple] = False
    
    # Collect all primes found by the sieve
    primes = [i for i, status in enumerate(is_prime) if status]

    count_case2 = 0
    
    # Iterate through p1 (primes[i])
    for i in range(len(primes)):
        p1 = primes[i]
        
        # Optimization: If p1 * p1 > M, then p1 * p2 (where p2 > p1) will definitely exceed M.
        # This is because p1 * p2 > p1 * p1.
        # So, no more valid pairs (p1, p2) can be formed.
        if p1 * p1 > M:
            break
        
        # Calculate the maximum possible p2 for this p1: p2 <= M / p1
        max_p2_for_p1 = M // p1
        
        # We need p2 > p1. So, we search for p2 in primes from index i+1 onwards.
        # If there are no primes after p1, or the next prime (primes[i+1]) is already too large, break.
        if i + 1 >= len(primes) or primes[i+1] > max_p2_for_p1:
            break
        
        # Use bisect_right to find the insertion point for max_p2_for_p1 in the sorted primes list.
        # bisect_right(primes, X) returns an index k such that all elements up to primes[k-1] are <= X.
        # So primes[k-1] is the largest prime <= X.
        idx_upper = bisect_right(primes, max_p2_for_p1)
        
        # The valid p2s are primes from index (i+1) up to (idx_upper - 1).
        # The number of such primes is (idx_upper - 1) - (i + 1) + 1 = idx_upper - (i + 1).
        num_p2s = idx_upper - (i + 1)
        
        if num_p2s > 0:
            count_case2 += num_p2s

    # Print the total count to standard output
    sys.stdout.write(str(count_case1 + count_case2) + '
')

solve()