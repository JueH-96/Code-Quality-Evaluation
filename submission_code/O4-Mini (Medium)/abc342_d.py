import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Count zeros separately
    cnt0 = A.count(0)

    # Prepare for sieve: find maximum A_i
    maxA = max(A) if A else 0
    if maxA < 2:
        # All A_i are 0 or 1 -> only zeros contribute pairs; 1*1=1 is square too
        # Actually, 1*1=1 is a square, so ones form pairs among themselves
        cnt1 = A.count(1)
        # zero with any is square, and ones among themselves
        total = cnt0*(N - cnt0) + cnt0*(cnt0 - 1)//2 + cnt1*(cnt1 - 1)//2
        print(total)
        return

    # Build smallest-prime-factor sieve up to maxA
    spf = list(range(maxA + 1))
    spf[0] = 0
    spf[1] = 1
    import math
    for i in range(2, int(math.isqrt(maxA)) + 1):
        if spf[i] == i:
            for j in range(i*i, maxA+1, i):
                if spf[j] == j:
                    spf[j] = i

    # For each non-zero A_i, compute its square-free core
    from collections import Counter
    core_count = Counter()
    for x in A:
        if x == 0:
            continue
        # factor x using spf
        orig = x
        core = 1
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            # keep p if exponent is odd
            if cnt & 1:
                core *= p
        core_count[core] += 1

    # Count pairs among non-zeros: same core gives perfect square product
    total = 0
    for cnt in core_count.values():
        total += cnt * (cnt - 1) // 2

    # Add pairs involving zero: zero with any other gives product 0 which is a square
    if cnt0 > 0:
        total += cnt0 * (N - cnt0)  # zero with non-zero
        total += cnt0 * (cnt0 - 1) // 2  # zero with zero

    print(total)

if __name__ == "__main__":
    main()