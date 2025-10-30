def main():
    import sys
    import random

    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    idx = 2

    A = list(map(int, data[idx: idx+N]))
    idx += N
    B = list(map(int, data[idx: idx+N]))
    idx += N

    # We'll use two independent 64-bit randoms per value to reduce collision probability
    mask = (1 << 64) - 1
    R1 = [random.getrandbits(64) & mask for _ in range(N+1)]
    R2 = [random.getrandbits(64) & mask for _ in range(N+1)]

    # Build prefix sums for A and B with each random array
    prefixA1 = [0] * (N+1)
    prefixA2 = [0] * (N+1)
    prefixB1 = [0] * (N+1)
    prefixB2 = [0] * (N+1)

    for i in range(1, N+1):
        prefixA1[i] = (prefixA1[i-1] + R1[A[i-1]]) & mask
        prefixA2[i] = (prefixA2[i-1] + R2[A[i-1]]) & mask
        prefixB1[i] = (prefixB1[i-1] + R1[B[i-1]]) & mask
        prefixB2[i] = (prefixB2[i-1] + R2[B[i-1]]) & mask

    ans = []
    for _ in range(Q):
        l, r, L, R = map(int, data[idx:idx+4])
        idx += 4

        # Subrange hash for A's [l..r]
        ha1 = (prefixA1[r] - prefixA1[l-1]) & mask
        ha2 = (prefixA2[r] - prefixA2[l-1]) & mask

        # Subrange hash for B's [L..R]
        hb1 = (prefixB1[R] - prefixB1[L-1]) & mask
        hb2 = (prefixB2[R] - prefixB2[L-1]) & mask

        ans.append("Yes" if (ha1 == hb1 and ha2 == hb2) else "No")

    print("
".join(ans))

# Don't forget to call main
if __name__ == "__main__":
    main()