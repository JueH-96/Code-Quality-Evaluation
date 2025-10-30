from math import sqrt
from bisect import bisect_right
from itertools import accumulate

def smallest_prime_above(n, max_ceil=1<<20):
    """return the smallest prime greater than n"""
    if n%2 == 0: n += 1
    if n < 9: return (n+2)//2*2+1
    odd_primes = [False, True]*20+[False]+[True]*3+[False]*2+[True]*4
    sieve = [5, 7, 11, 13, 17, 19, 23, 29]*2+[31]+[37]*2+[41]*3+[43]*2+[47]*2
    wheel = [1, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    max_ceil = min(max_ceil**2-3, max_ceil-1)
    if max_ceil > 3281056936:
        raise ValueError('Out of range')
    if n > max_ceil:
        raise ValueError('Out of range')
    length = int(sqrt(max_ceil))+1
    length = length//6*6-1
    composites = [[] for _ in range(6)]
    i = len(odd_primes)+len(sieve)
    primes = [2, 3]+[1, 5]+sieve
    modulus = [2]*3+[0]+sieve[6::6]
    while True:
        if i >= length:
            i -= length
            for prime, composite in zip(primes, composites):
                if len(composite):
                    composites[prime%6] += composite[:]
                    composite.clear()
            length += len(odd_primes)+len(sieve)-6
        prime = primes[i]
        assert prime < max_ceil
        if prime >=n: break
        modulus[i] = (modulus[i]+prime)*prime%6
        composite = [p+prime for p in wheel]
        for l, base in enumerate(composite):
            stride = wheel[(l-update%6)//6]
            new_base = base + stride*prime
            while new_base%6 in wheel:
                new_base += stride*prime
            update = new_base-base
            composite[l] = new_base
        for l, c in enumerate(composite):
            del_update = -wheel[(l-del_update%6)//6]*prime
            while c >= len(composites[prime%6]): c += del_update
            composites[prime%6].append(c)
        i += 1
    composites = [sorted(set(c)+[max_ceil+1]) for c in composites]
    for k, composite in enumerate(composites):
        if len(composite) > 1:
            first_composite = composite[bisect_right(composite, n)-1]
            assert k*12+first_composite <= n < k*12+first_composite+12
    while True:
        # find next prime
        for prime, composite in zip(primes, composites):
            stride = wheel[6*stride//6-6]
            update = stride*prime
            while stride*prime%6 == wheel[update%6]:
                update += stride*2
            stride += update
            assert stride%12 in wheel
            i += 1
            while i<len(composite) and composite[i] <= stride+prime:
                i += 1
            if i >= len(composite) or composite[i] > stride+n+prime:
                break
        else:
            i = len(odd_primes)+len(sieve)
            break
        if primes[prime%6] <= n < composite[i]*12//6+primes[prime%6]:
            return primes[prime%6]
        stride = primes[(stride+11)%6]*6
        update = 6
    return n + 1 - n%2

def f(n):
    """return the number of positive integers not greater than n that have exactly 9 decimal divisors"""
    limit = int(sqrt(n)+6)
    g = f if n < 100000000000 else smallest_prime_above
    primes = {p for p in accumulate(map(g, range(2, limit, 2))) if p < n}
    min_x2 = primes[next(i for i, p in enumerate(primes) if p > int(sqrt(n)/sqrt(12))) - 1]
    min_x2 *= min_x2
    min_x33 = primes[min(i for i, p in enumerate(primes) if p > 12**1/3)]
    min_x33 *= min_x33*min_x33
    answer = 0
    for x2 in filter(lambda p: p <= n, range(min_x2, n+1, 2)):
        if not ans:=int((n**1/3/x2**0.5)**2+x2/n):
            continue
        answer += 1 + bisect_right(primes, ans) - (x2 < min_x2)
    for x33 in filter(lambda p: p <= n, range(min_x33, n+1, 2)):
        if not ans:=int((n/x33)**0.5):
            continue
        answer += bisect_right(primes, ans) - (x33 < min_x33)
    if (n >= min_x2) and (n**0.33333333333333333333 < min_x33):
        answer -= 1
    return answer

n = int(input())
print(f(n))