import math

def mobius(n):
    if n == 1:
        return 1
    factors = {}
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            if factors[i] > 1:
                return 0
            n //= i
        i += 1
    if n > 1:
        factors[n] = 1
    return (-1) ** len(factors)

def power(a, b, limit):
    result = 1
    while b > 0:
        if b % 2 == 1:
            if result > limit // a:
                return limit + 1
            result *= a
        a_sq = a
        for _ in range(b // 2):
            a_sq *= a_sq
            if a_sq > limit:
                break
        if a_sq > limit:
            return limit + 1
        b //= 2
    return result

def compute_c(a_max, k):
    if a_max < 1:
        return 0
    low = 1
    high = a_max
    c = 0
    while low <= high:
        mid = (low + high) // 2
        p = power(mid, k, a_max)
        if p <= a_max:
            c = mid
            low = mid + 1
        else:
            high = mid - 1
    return c

def main():
    import sys
    N = int(sys.stdin.readline())
    if N < 1:
        print(0)
        return
    count = 1  # for x=1
    if N == 0:
        print(0)
        return
    b_max = N.bit_length() - 1
    for b in range(2, b_max + 1):
        a_low = 2
        a_high = N
        a_max = 1
        while a_low <= a_high:
            mid = (a_low + a_high) // 2
            p = power(mid, b, N)
            if p <= N:
                a_max = mid
                a_low = mid + 1
            else:
                a_high = mid - 1
        if a_max < 2:
            continue
        sum_k = 0
        max_k = a_max.bit_length() - 1
        for k in range(2, max_k + 1):
            c = compute_c(a_max, k)
            mu = mobius(k)
            if mu != 0:
                sum_k += (c - 1) * mu
        non_perfect = (a_max - 1) - sum_k
        count += non_perfect
    print(count)

if __name__ == "__main__":
    main()