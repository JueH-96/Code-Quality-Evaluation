import sys
import math

def compute_mobius(max_d):
    mobius = [1] * (max_d + 1)
    is_prime = [True] * (max_d + 1)
    for p in range(2, max_d + 1):
        if is_prime[p]:
            for multiple in range(p, max_d + 1, p):
                is_prime[multiple] = False
                mobius[multiple] *= -1
            p_square = p * p
            for multiple in range(p_square, max_d + 1, p_square):
                mobius[multiple] = 0
    return mobius

# Precompute Möbius function up to 3e4
max_mobius = 30000
mobius = compute_mobius(max_mobius)

def count_square_free(M):
    if M == 0:
        return 0
    res = 0
    max_d = int(math.isqrt(M))
    for d in range(1, max_d + 1):
        if mobius[d] == 0:
            continue
        term = M // (d * d)
        res += mobius[d] * term
    return res

def compute_M_e(N, e):
    if e == 0:
        return 0
    low = 1
    high = N
    best = 0
    while low <= high:
        mid = (low + high) // 2
        res = 1
        overflow = False
        for _ in range(e):
            res *= mid
            if res > N:
                overflow = True
                break
        if overflow:
            high = mid - 1
        else:
            best = mid
            low = mid + 1
    return best

def main():
    N = int(sys.stdin.readline())
    if N < 2:
        print(0)
        return

    max_e = 1
    while (max_e + 1) ** 2 <= N:
        max_e += 1
    # Binary search to find maximum e where 2^e <= N
    low = 2
    high = 60  # since 2^60 is about 1e18
    max_e = 1
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            high = mid - 1
            continue
        try:
            val = 2 ** mid
            if val <= N:
                max_e = mid
                low = mid + 1
            else:
                high = mid - 1
        except OverflowError:
            high = mid - 1

    sum_e = 0
    for e in range(2, max_e + 1):
        M_e = compute_M_e(N, e)
        if M_e == 0:
            continue
        count = count_square_free(M_e)
        sum_e += count

    total = sum_e - (max_e - 2)
    print(total)

if __name__ == "__main__":
    main()