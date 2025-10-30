import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [0] * (n + 1)
    B = [0] * (n + 1)
    for i in range(1, n + 1):
        A[i] = int(next(it))
    for i in range(1, n + 1):
        B[i] = int(next(it))
    # Assign two independent 64-bit random hashes for each possible value 1..n
    import random
    mask = (1 << 64) - 1
    h1 = [random.getrandbits(64) & mask for _ in range(n + 1)]
    h2 = [random.getrandbits(64) & mask for _ in range(n + 1)]
    # Build prefix sums (mod 2^64) of hashes for A and B
    HA1 = [0] * (n + 1)
    HA2 = [0] * (n + 1)
    HB1 = [0] * (n + 1)
    HB2 = [0] * (n + 1)
    for i in range(1, n + 1):
        HA1[i] = (HA1[i-1] + h1[A[i]]) & mask
        HA2[i] = (HA2[i-1] + h2[A[i]]) & mask
        HB1[i] = (HB1[i-1] + h1[B[i]]) & mask
        HB2[i] = (HB2[i-1] + h2[B[i]]) & mask

    out = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R = int(next(it))
        sumA1 = (HA1[r] - HA1[l-1]) & mask
        sumA2 = (HA2[r] - HA2[l-1]) & mask
        sumB1 = (HB1[R] - HB1[L-1]) & mask
        sumB2 = (HB2[R] - HB2[L-1]) & mask
        if sumA1 == sumB1 and sumA2 == sumB2:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()