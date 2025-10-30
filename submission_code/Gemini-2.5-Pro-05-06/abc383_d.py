import math
import bisect

def solve():
    N = int(input())

    # Step 1: Sieve primes up to a reasonable limit.
    # Max value for p1*p2 is sqrt(N). Max sqrt(N) is sqrt(4e12) = 2e6.
    # So, primes up to 2e6 are needed.
    SIEVE_LIMIT = 2 * 10**6 
    
    is_prime_arr = [True] * (SIEVE_LIMIT + 1)
    # 0 and 1 are not prime.
    if SIEVE_LIMIT >= 0: is_prime_arr[0] = False
    if SIEVE_LIMIT >= 1: is_prime_arr[1] = False
    
    for p_sieve in range(2, int(math.sqrt(SIEVE_LIMIT)) + 1):
        if is_prime_arr[p_sieve]:
            for multiple in range(p_sieve * p_sieve, SIEVE_LIMIT + 1, p_sieve):
                is_prime_arr[multiple] = False
    
    primes_list = []
    for p_sieve in range(2, SIEVE_LIMIT + 1):
        if is_prime_arr[p_sieve]:
            primes_list.append(p_sieve)

    ans_count = 0

    # Step 2: Case 1: Numbers of the form x = p^8
    count1 = 0
    for p_val_case1 in primes_list:
        # p_val_case1^8 must be <= N.
        # Python handles large integers, so direct computation of p^8 is fine.
        # Max p_val_case1 for N=4e12 is 37. 37^8 fits in a 64-bit integer.
        # The loop will break quickly if p_val_case1^8 exceeds N.
        
        # A check to prevent pow from attempting very large exponents if p_val_case1 is large,
        # though this is implicitly handled by N. Max p here is 37.
        # For p_val_case1 = 41, 41^8 > 4e12. Loop breaks.
        # If p_val_case1 is already such that p_val_case1^2 itself is huge, p_val_case1^8 would be even larger.
        # e.g. if p_val_case1 = 100, then p_val_case1^8 is (100)^8 = 10^16.
        # This might happen if N is very small. E.g., N=10.
        # Then for p_val_case1=2, 2^8=256. 256 > 10, so loop breaks. This is efficient.
        
        val_p_pow_8 = pow(p_val_case1, 8)

        if val_p_pow_8 <= N:
            count1 += 1
        else:
            # Since primes_list is sorted, further primes will also yield p^8 > N.
            break 
    ans_count += count1

    # Step 3: Case 2: Numbers of the form x = p1^2 * p2^2 = (p1*p2)^2
    # We need p1*p2 <= sqrt(N). Let M_sqrt_N = floor(sqrt(N)).
    # So, p1*p2 <= M_sqrt_N, with p1 < p2 primes.
    
    if N == 0: # Constraint N >= 1, so N cannot be 0.
        M_sqrt_N = 0
    else:
        # math.isqrt is preferred for potentially large N (Python 3.8+).
        if hasattr(math, 'isqrt'):
            M_sqrt_N = math.isqrt(N)
        else:
            M_sqrt_N = int(N**0.5) # Works for N up to 4e12.

    count2 = 0
    num_total_primes = len(primes_list)

    for i in range(num_total_primes):
        p1 = primes_list[i]
        
        # For p1*p2 <= M_sqrt_N with p1 < p2:
        # p1 must be < sqrt(M_sqrt_N). So, p1*p1 < M_sqrt_N.
        # If p1*p1 >= M_sqrt_N, then p1*p2 > p1*p1 >= M_sqrt_N. So, break.
        if p1 * p1 >= M_sqrt_N:
            break

        # p2 must satisfy: p1 < p2 <= M_sqrt_N / p1.
        upper_bound_for_p2_val = M_sqrt_N // p1
        
        # Smallest possible p2 is primes_list[i+1].
        # If this smallest p2 is already greater than upper_bound_for_p2_val, no suitable p2 exists.
        if i + 1 >= num_total_primes: # p1 is the last prime in list, no p2 > p1.
            break 
        if primes_list[i+1] > upper_bound_for_p2_val:
            continue # Current p1 cannot form a pair. Move to next p1.
        
        # Count primes p_k in primes_list such that:
        # primes_list[i] < p_k <= upper_bound_for_p2_val.
        # The search for p_k starts from index i+1.
        start_idx_for_p2_search = i + 1
        
        # bisect_right(a, x, lo, hi) finds insertion point for x in a[lo:hi].
        # The returned index is w.r.t. the original list 'a'.
        # Elements a[start_idx_for_p2_search ... effective_end_idx_for_p2-1] are <= upper_bound_for_p2_val.
        effective_end_idx_for_p2 = bisect.bisect_right(primes_list, 
                                                       upper_bound_for_p2_val, 
                                                       lo=start_idx_for_p2_search, 
                                                       hi=num_total_primes)
        
        num_p2_found = effective_end_idx_for_p2 - start_idx_for_p2_search
        
        if num_p2_found > 0:
            count2 += num_p2_found
            
    ans_count += count2
    
    print(ans_count)

solve()