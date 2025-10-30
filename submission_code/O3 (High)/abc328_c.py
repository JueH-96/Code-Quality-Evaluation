import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    # data: [N, Q, S, l1, r1, ...]
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    S = next(it).decode()

    # Build prefix sum of "adjacent equal" indicators.
    # P[i] = number of positions p with S_p == S_{p+1} in range [1, i]
    # (1-based indexing; P[0] = 0).
    P = [0] * (N)  # size N, indices 0..N-1
    for i in range(1, N):
        P[i] = P[i - 1] + (1 if S[i - 1] == S[i] else 0)

    out_lines = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        # count positions p in [l, r-1] with S_p == S_{p+1}
        ans = P[r - 1] - P[l - 1]
        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()