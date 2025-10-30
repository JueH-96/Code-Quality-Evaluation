import math
import bisect

mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    Q = int(math.isqrt(N))
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, n + 1) if is_prime[i]]
        return primes
    
    small_primes = sieve(Q)
    
    max_e = 100
    max_n = M - 1 + max_e
    if max_n < 0:
        max_n = 0
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
    for i in range(max_n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod
        
    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        res = fact[n] * inv_fact[r] % mod
        res = res * inv_fact[n - r] % mod
        return res
    
    B = [0] * (max_e + 1)
    for e in range(0, max_e + 1):
        B[e] = nCr(e + M - 1, M - 1)
    
    dp_bad = {1: 1}
    for p in small_primes:
        new_dp = dict(dp_bad)
        pe = p
        for e in range(1, max_e + 1):
            if pe > N:
                break
            allowed = True
            if p == 3:
                pass
            elif p % 3 == 1:
                if e % 3 == 2:
                    allowed = False
            else:
                if e % 2 == 1:
                    allowed = False
            if not allowed:
                pe *= p
                if pe > N:
                    break
                continue
            items = list(dp_bad.items())
            for m, count_val in items:
                m_new = m * pe
                if m_new > N:
                    continue
                new_dp[m_new] = (new_dp.get(m_new, 0) + count_val * B[e]) % mod
            pe *= p
            if pe > N:
                break
        dp_bad = new_dp

    dp_total = {1: 1}
    for p in small_primes:
        new_dp = dict(dp_total)
        pe = p
        for e in range(1, max_e + 1):
            if pe > N:
                break
            items = list(dp_total.items())
            for m, count_val in items:
                m_new = m * pe
                if m_new > N:
                    continue
                new_dp[m_new] = (new_dp.get(m_new, 0) + count_val * B[e]) % mod
            pe *= p
            if pe > N:
                break
        dp_total = new_dp

    distinct_x = set()
    for m in dp_bad:
        x = N // m
        distinct_x.add(x)
    for m in dp_total:
        x = N // m
        distinct_x.add(x)
    sorted_x = sorted(distinct_x)
    
    SEG = 10**6
    segments = []
    low = Q + 1
    high = N
    while low <= high:
        seg_high = min(low + SEG - 1, high)
        segments.append((low, seg_high))
        low = seg_high + 1
        
    base_primes = small_primes
    
    def sieve_segment(L, R, residue_type):
        size = R - L + 1
        sieve_arr = [True] * size
        for p in base_primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            if start < L:
                start += p
            if start > R:
                continue
            start_idx = start - L
            for j in range(start_idx, size, p):
                sieve_arr[j] = False
        primes_in_seg = []
        for i in range(size):
            if sieve_arr[i]:
                num = L + i
                if num < 2:
                    continue
                if residue_type == 'bad':
                    if num == 3 or (num % 3 == 1):
                        primes_in_seg.append(num)
                else:
                    primes_in_seg.append(num)
        return primes_in_seg

    results_bad = {}
    global_count_bad = 0
    for seg in segments:
        L_seg, R_seg = seg
        primes_in_seg = sieve_segment(L_seg, R_seg, 'bad')
        xs_in_seg = [x for x in sorted_x if L_seg <= x <= R_seg]
        xs_in_seg.sort()
        for x in xs_in_seg:
            pos = bisect.bisect_right(primes_in_seg, x)
            count_here = pos
            results_bad[x] = global_count_bad + count_here
        global_count_bad += len(primes_in_seg)
        
    results_total = {}
    global_count_total = 0
    for seg in segments:
        L_seg, R_seg = seg
        primes_in_seg = sieve_segment(L_seg, R_seg, 'total')
        xs_in_seg = [x for x in sorted_x if L_seg <= x <= R_seg]
        xs_in_seg.sort()
        for x in xs_in_seg:
            pos = bisect.bisect_right(primes_in_seg, x)
            count_here = pos
            results_total[x] = global_count_total + count_here
        global_count_total += len(primes_in_seg)
        
    F_bad = 0
    for m, count_val in dp_bad.items():
        x = N // m
        cnt_large = results_bad.get(x, 0)
        term = count_val * (1 + M * cnt_large) % mod
        F_bad = (F_bad + term) % mod
        
    TotalCount = 0
    for m, count_val in dp_total.items():
        x = N // m
        cnt_large = results_total.get(x, 0)
        term = count_val * (1 + M * cnt_large) % mod
        TotalCount = (TotalCount + term) % mod
        
    ans = (TotalCount - F_bad) % mod
    print(ans)

if __name__ == "__main__":
    main()