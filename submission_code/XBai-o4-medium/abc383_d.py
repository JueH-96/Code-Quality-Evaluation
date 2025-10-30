import math
import bisect

def sieve(n):
    if n < 2:
        return []
    sieve_list = [True] * (n + 1)
    sieve_list[0] = sieve_list[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve_list[i]:
            sieve_list[i*i : n+1 : i] = [False] * len(sieve_list[i*i : n+1 : i])
    primes = [i for i, is_prime in enumerate(sieve_list) if is_prime]
    return primes

def find_max_p8(N):
    if N < 2**8:
        return 1
    low = 2
    high = 2
    while (high ** 8) <= N:
        high *= 2
    ans = 1
    while low <= high:
        mid = (low + high) // 2
        m8 = mid ** 8
        if m8 <= N:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    return ans

def main():
    N = int(input().strip())
    # Case 1: p^8
    max_p8 = find_max_p8(N)
    primes_case1 = sieve(max_p8)
    count_case1 = len(primes_case1)
    # Case 2: (p*q)^2 <= N → p*q <= K = sqrt(N)
    K = math.isqrt(N)
    primes_case2 = sieve(K)
    count_case2 = 0
    len_primes = len(primes_case2)
    for i in range(len_primes):
        p = primes_case2[i]
        if p * p > K:
            break
        x = K // p
        # find number of primes in primes_case2 <=x
        num_le_x = bisect.bisect_right(primes_case2, x)
        current_count = num_le_x - i - 1
        if current_count > 0:
            count_case2 += current_count
    total = count_case1 + count_case2
    print(total)

if __name__ == "__main__":
    main()