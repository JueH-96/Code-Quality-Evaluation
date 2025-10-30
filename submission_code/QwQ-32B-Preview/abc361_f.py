import math

def mobius(b, factors):
    if b == 1:
        return 1
    if b in factors:
        return factors[b]
    original_b = b
    factors[b] = 0  # temporary placeholder
    mu = 1
    for i in range(2, int(math.isqrt(b)) + 1):
        if b % i == 0:
            count = 0
            while b % i == 0:
                b //= i
                count += 1
            if count > 1:
                return 0
            mu *= -1
    if b > 1:
        mu *= -1
    factors[original_b] = mu
    return mu

def integer_root(N, b):
    if N == 0:
        return 0
    low = 0
    high = N
    while low < high:
        mid = (low + high + 1) // 2
        if mid ** b > N:
            high = mid - 1
        else:
            low = mid
    return low

def count_perfect_powers(N):
    if N < 1:
        return 0
    max_b = 1
    while (1 << max_b) <= N:
        max_b += 1
    factors = {}
    total = 0
    for b in range(2, max_b + 1):
        mu_b = mobius(b, factors)
        if mu_b == 0:
            continue
        count_b = integer_root(N, b)
        total += mu_b * count_b
    return total

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    print(count_perfect_powers(N))

if __name__ == "__main__":
    main()