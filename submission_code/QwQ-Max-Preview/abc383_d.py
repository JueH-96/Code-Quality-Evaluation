import math
import bisect

def sieve(n):
    if n < 2:
        return []
    sieve_list = [True] * (n + 1)
    sieve_list[0], sieve_list[1] = False, False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve_list[i]:
            sieve_list[i*i : n+1 : i] = [False] * len(sieve_list[i*i : n+1 : i])
    return [i for i, is_prime in enumerate(sieve_list) if is_prime]

def find_max_p_case1(N):
    if N < 1:
        return 0
    low, high = 0, N
    max_p = 0
    while low <= high:
        mid = (low + high) // 2
        try:
            mid_pow8 = mid ** 8
        except OverflowError:
            high = mid - 1
            continue
        if mid_pow8 <= N:
            max_p = mid
            low = mid + 1
        else:
            high = mid - 1
    return max_p

def main():
    N = int(input().strip())
    if N < 1:
        print(0)
        return
    
    # Case 1: Numbers of the form p^8
    max_p_case1 = find_max_p_case1(N)
    primes_case1 = sieve(max_p_case1)
    count_case1 = len(primes_case1)
    
    # Case 2: Numbers of the form p²*q² where p < q
    s = math.isqrt(N)
    primes_case2 = sieve(s)
    sqrt_s = math.isqrt(s)
    count_case2 = 0
    
    for p in primes_case2:
        if p > sqrt_s:
            break
        max_q = s // p
        if max_q <= p:
            continue
        # Find the count of primes q > p and <= max_q
        q_max_idx = bisect.bisect_right(primes_case2, max_q) - 1
        if q_max_idx < 0:
            continue
        p_idx = bisect.bisect_right(primes_case2, p)
        if q_max_idx >= p_idx:
            count_case2 += q_max_idx - p_idx + 1
    
    total = count_case1 + count_case2
    print(total)

if __name__ == '__main__':
    main()