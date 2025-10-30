import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = list(map(int, (next(it) for _ in range(n))))
    maxA = max(A)

    # Build smallest prime factor (spf) sieve up to maxA
    spf = list(range(maxA+1))
    spf[0] = 0
    if maxA >= 1:
        spf[1] = 1
    import math
    lim = int(math.isqrt(maxA))
    for i in range(2, lim+1):
        if spf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, maxA+1, step):
                if spf[j] == j:
                    spf[j] = i

    # Compute the XOR of Grundy values = total prime-multiplicity for each A_i
    xorsum = 0
    for a in A:
        cnt = 0
        v = a
        # count prime factors with multiplicity using spf
        while v > 1:
            p = spf[v]
            v //= p
            cnt += 1
        xorsum ^= cnt

    # If xor != 0, first player (Anna) wins; else Bruno wins
    print("Anna" if xorsum != 0 else "Bruno")

if __name__ == "__main__":
    main()