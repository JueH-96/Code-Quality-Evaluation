import math

def is_square_free(k):
    if k == 1:
        return True
    for p in range(2, int(math.isqrt(k)) + 1):
        if p * p > k:
            break
        if k % (p * p) == 0:
            return False
    sqrt_k = int(math.isqrt(k))
    if sqrt_k * sqrt_k == k:
        return False
    return True

def mobius(k):
    if not is_square_free(k):
        return 0
    factors = set()
    temp = k
    for p in range(2, int(math.isqrt(temp)) + 1):
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return -1 if len(factors) % 2 else 1

def find_a_max(k, N):
    if k == 0:
        return 0
    if N < (1 << k):
        return 1
    low = 2
    high = N
    a_max = 1
    while low <= high:
        mid = (low + high) // 2
        current = 1
        overflow = False
        for _ in range(k):
            current *= mid
            if current > N:
                overflow = True
                break
        if overflow:
            high = mid - 1
        else:
            a_max = mid
            low = mid + 1
    return a_max

def main():
    N = int(input())
    if N == 0:
        print(0)
        return
    elif N == 1:
        print(1)
        return
    total = 0
    max_k = 60
    for k in range(2, max_k + 1):
        if 2 ** k > N:
            continue
        if not is_square_free(k):
            continue
        mu = mobius(k)
        if mu == 0:
            continue
        a_max = find_a_max(k, N)
        m = a_max - 1
        if m < 0:
            m = 0
        total += (-mu) * m
    print(1 + total)

if __name__ == "__main__":
    main()