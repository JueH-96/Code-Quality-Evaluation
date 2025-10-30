import bisect

def sieve(n):
    if n < 2:
        return []
    sieve_list = [True] * (n + 1)
    sieve_list[0] = sieve_list[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve_list[i]:
            sieve_list[i*i : n+1 : i] = [False] * len(sieve_list[i*i : n+1 : i])
    primes = [i for i, is_prime in enumerate(sieve_list) if is_prime]
    return primes

def find_p_max(N):
    if N < 1:
        return 0
    low = 2
    high = 2
    while high ** 8 <= N:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if mid ** 8 <= N:
            low = mid + 1
        else:
            high = mid
    return low - 1

def main():
    import sys
    N = int(sys.stdin.readline())
    
    # Case 1: x = p^8
    p_max = find_p_max(N)
    primes_p = sieve(p_max)
    count_case1 = len(primes_p)
    
    # Case 2: x = p^2 * q^2 where p < q
    K = int(N ** 0.5)
    primes_q = sieve(K)
    count_case2 = 0
    for i, p in enumerate(primes_q):
        if p > K // p:
            continue
        q_max = K // p
        j = bisect.bisect_right(primes_q, q_max) - 1
        if j < i:
            continue
        count_case2 += (j - i)
    
    total = count_case1 + count_case2
    print(total)

if __name__ == '__main__':
    main()