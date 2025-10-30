import sys
import bisect

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    # Number of villages
    N = next(it)

    # Coordinates and populations
    X = [next(it) for _ in range(N)]
    P = [next(it) for _ in range(N)]

    # Prefix sums of populations
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + P[i]

    # Number of queries
    Q = next(it)

    out_lines = []
    for _ in range(Q):
        L = next(it)
        R = next(it)

        # First index with coordinate >= L
        left_idx = bisect.bisect_left(X, L)
        # First index with coordinate  > R   (upper bound)
        right_idx = bisect.bisect_right(X, R)

        # Sum between indices [left_idx, right_idx)
        villagers = prefix[right_idx] - prefix[left_idx]
        out_lines.append(str(villagers))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()