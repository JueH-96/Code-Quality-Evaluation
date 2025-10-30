import sys
import random

def main():
    data = sys.stdin.readline().split()
    N = int(data[0]); Q = int(data[1])
    A_vals = list(map(int, sys.stdin.readline().split()))
    B_vals = list(map(int, sys.stdin.readline().split()))

    # Prepare two independent 64-bit random hash values for each possible integer
    H1 = [random.getrandbits(64) for _ in range(N+1)]
    H2 = [random.getrandbits(64) for _ in range(N+1)]
    mask = (1 << 64) - 1

    # Build prefix‐sum hashes for A and for B
    SA1 = [0] * (N+1)
    SA2 = [0] * (N+1)
    SB1 = [0] * (N+1)
    SB2 = [0] * (N+1)
    for i, v in enumerate(A_vals, start=1):
        SA1[i] = (SA1[i-1] + H1[v]) & mask
        SA2[i] = (SA2[i-1] + H2[v]) & mask
    for i, v in enumerate(B_vals, start=1):
        SB1[i] = (SB1[i-1] + H1[v]) & mask
        SB2[i] = (SB2[i-1] + H2[v]) & mask

    out = []
    rd = sys.stdin.readline
    for _ in range(Q):
        l, r, L0, R0 = map(int, rd().split())
        # Quick length check
        if r - l != R0 - L0:
            out.append("No")
            continue
        # Compute the hash of A[l..r] and B[L0..R0]
        ha1 = (SA1[r]   - SA1[l-1]) & mask
        ha2 = (SA2[r]   - SA2[l-1]) & mask
        hb1 = (SB1[R0]  - SB1[L0-1]) & mask
        hb2 = (SB2[R0]  - SB2[L0-1]) & mask
        # If both hash‐pairs match, we assume multisets match
        out.append("Yes" if (ha1 == hb1 and ha2 == hb2) else "No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()