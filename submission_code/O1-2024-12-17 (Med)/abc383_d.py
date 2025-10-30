def main():
    import sys
    import math
    import bisect
    
    input_data = sys.stdin.read().strip()
    N = int(input_data)
    
    # Step 1: Generate all primes up to sqrt(N)
    limit = math.isqrt(N)  # largest integer p such that p^2 <= N
    sieve = [True]*(limit+1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5)+1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    primes = [x for x in range(2, limit+1) if sieve[x]]
    
    # Step 2: Count how many primes p satisfy p^8 <= N
    # Safely compute the integer 8th root of N via binary search
    def eighth_root(x):
        lo, hi = 0, 1000000  # more than enough upper bound
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if mid**8 <= x:
                lo = mid
            else:
                hi = mid - 1
        return lo
    
    r8 = eighth_root(N)
    # Number of primes <= r8
    count_p8 = bisect.bisect_right(primes, r8)
    
    # Step 3: Count integers of the form p^2 * q^2 <= N with p < q distinct primes
    count_p2q2 = 0
    for i, p in enumerate(primes):
        # if p^2 > N, we can break early
        if p*p > N:
            break
        # We want q^2 <= N // (p^2)
        Nx = N // (p*p)
        if Nx == 0:
            break
        up = math.isqrt(Nx)
        # We only count q where q > p and q <= up
        if up <= p:
            continue
        idx_up = bisect.bisect_right(primes, up) - 1
        if idx_up > i:  # ensures q > p
            count_p2q2 += (idx_up - i)
    
    # Final answer is the sum of counts
    print(count_p8 + count_p2q2)

# Don't forget to call main()
main()