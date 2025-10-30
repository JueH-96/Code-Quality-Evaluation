#!/usr/bin/env python3
import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    Q = int(next(it))
    As = [int(next(it)) for _ in range(Q)]
    MAX_A = 10**12

    # Sieve primes up to limit for q: need primes up to sqrt(MAX_A/4) ~ 5e5
    LIM = 500000
    sieve = bytearray(b'\x01') * (LIM+1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(LIM**0.5) + 1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:LIM+1:step] = b'\x00' * (((LIM - start)//step) + 1)
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    n_pr = len(primes)

    # Precompute powers p^2, p^4, ... upto reasonable cap for all primes
    # But for p in p-loop, we'll cap based on next_prime
    # For q, we need pow_q too; compute full pow_list up to MAX_A/4
    max_pow_base = MAX_A // 4
    pow_list = {}
    for p in primes:
        pw = p * p
        if pw > max_pow_base:
            break
        lst = []
        # step exponents by 2: multiply by p*p each time
        while pw <= max_pow_base:
            lst.append(pw)
            # watch for overflow
            # next pw = pw * p * p
            # but if pw > max_pow_base//(p*p): break
            if pw > max_pow_base // (p*p):
                break
            pw *= p*p
        pow_list[p] = lst

    # Generate all "400-numbers": N = p^a * q^b, a,b even >=2, p<q primes, N<=MAX_A
    vals = set()
    import bisect

    for i, p in enumerate(primes):
        # need next prime q exists
        if i+1 >= n_pr:
            break
        q0 = primes[i+1]
        # break if even minimal p^2*q0^2 > MAX_A
        if p*p * q0*q0 > MAX_A:
            break
        # get pow_p list for this p
        lst_p = pow_list.get(p)
        if not lst_p:
            continue
        # For this p, cap each power by MAX_A/(q0^2)
        cap_p = MAX_A // (q0*q0)
        # iterate pow_p values up to cap_p
        for pow_p in lst_p:
            if pow_p > cap_p:
                break
            # for this pow_p, find q primes >p with q^2 <= MAX_A/pow_p
            lim = MAX_A // pow_p
            # we need q^2 <= lim -> q <= sqrt(lim)
            # compute integer sqrt
            maxq = int(lim**0.5)
            # find index of first prime > p
            j0 = i+1
            # find index of last prime <= maxq
            # primes list is sorted
            j1 = bisect.bisect_right(primes, maxq) - 1
            if j1 < j0:
                continue
            # iterate q indices
            for j in range(j0, j1+1):
                q = primes[j]
                # get pow_q list
                lst_q = pow_list.get(q)
                if not lst_q:
                    continue
                # for each pow_q, pow_p * pow_q <= MAX_A
                # since lst_q is sorted increasing, break when too large
                for pow_q in lst_q:
                    prod = pow_p * pow_q
                    if prod > MAX_A:
                        break
                    # ensure distinct primes mandatory since q>p
                    vals.add(prod)

    # Convert to sorted list
    arr = sorted(vals)

    # For each query, binary search largest <= A
    out = []
    import bisect
    for A in As:
        idx = bisect.bisect_right(arr, A) - 1
        # as per problem statement, answer always exists
        out.append(str(arr[idx]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()