import sys

def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    # A[0]=0 < A[1] < ... < A[N-1] <= 1e18

    # We choose a horizon H beyond A[N-1].  Empirically H=200_000 is enough.
    H = 200_000
    maxA = A[-1]
    M = min((1 << 20), maxA + H)  # clip at 2^20 ~ 1e6

    # Build paper-folding sequence g[1..M], where
    #   g(n)=1 if n-th crease is mountain, 0 if valley.
    G = [0] * (M + 1)
    for n in range(1, M + 1):
        if n & 1:
            # odd: n=2m+1
            if (n & 3) == 3:
                G[n] = 1
            else:
                G[n] = 0
        else:
            G[n] = G[n >> 1]

    # Now slide i from 1..(M - maxA) and compute sum_{k=1..N} G[i + A_k]
    # Keep the global maximum.
    best = 0
    limit = M - maxA
    # Precompute G at the offsets when shifted by i=1
    window_sum = 0
    for k in range(N):
        window_sum += G[1 + A[k]]
    best = window_sum

    for i in range(2, limit + 1):
        # Remove the old leftmost, add the new rightmost
        window_sum -= G[i - 1 + A[0]]     # A[0] == 0, so this is G[i-1]
        window_sum += G[i + A[N - 1]]
        if window_sum > best:
            best = window_sum

    print(best)


if __name__ == "__main__":
    main()