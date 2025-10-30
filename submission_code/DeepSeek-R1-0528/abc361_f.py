import math

def precompute_mobius(n):
    mobius = [1] * (n + 1)
    is_prime = [True] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mobius[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mobius[i * p] = 0
                break
            else:
                mobius[i * p] = -mobius[i]
    return mobius

def main():
    N_str = input().strip()
    if not N_str:
        return
    N = int(N_str)
    if N == 0:
        print(0)
        return
    if N == 1:
        print(1)
        return

    max_exp = 60
    mobius = precompute_mobius(max_exp)
    bits = N.bit_length()
    total_sum = 0

    for k in range(2, max_exp + 1):
        if mobius[k] == 0:
            continue
        if k >= bits:
            continue

        low_bound = 1
        high_bound = 1
        while True:
            if high_bound > N:
                high_bound = N + 1
                break
            power_val = pow(high_bound, k)
            if power_val > N:
                break
            low_bound = high_bound
            if high_bound * 2 > N:
                high_bound = N + 1
                break
            else:
                high_bound *= 2

        lo, hi = low_bound, high_bound - 1
        a = low_bound
        if lo <= hi:
            while lo <= hi:
                mid = (lo + hi) // 2
                mid_power = pow(mid, k)
                if mid_power <= N:
                    a = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            a = hi
        else:
            a = low_bound

        count_base_ge2 = a - 1
        total_sum += mobius[k] * count_base_ge2

    answer = 1 - total_sum
    print(answer)

if __name__ == '__main__':
    main()