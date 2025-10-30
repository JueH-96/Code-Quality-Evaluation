import sys
import bisect

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    N = next(it)
    M = next(it)
    P = next(it)

    A = [next(it) for _ in range(N)]
    B = [next(it) for _ in range(M)]

    # sort side–dish prices and build prefix sums
    B.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + B[i]

    total = 0
    for a in A:
        limit = P - a
        if limit < 0:
            cnt = 0                # no B_j makes the sum ≤ P
        else:
            cnt = bisect.bisect_right(B, limit)   # # of B_j with A_i + B_j ≤ P

        # pairs with sum ≤ P:    cnt * a  +  sum of those B_j
        # pairs with sum >  P:   (M - cnt) * P
        total += cnt * a + prefix[cnt] + (M - cnt) * P

    print(total)

if __name__ == "__main__":
    main()