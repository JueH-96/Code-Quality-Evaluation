import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    # read input
    N, M, L = map(int, input().split())
    a = list(map(int, input().split()))          # prices of main dishes
    b = list(map(int, input().split()))          # prices of side dishes

    # store forbidden pairs for every main dish
    forbidden = [set() for _ in range(N)]
    for _ in range(L):
        c, d = map(int, input().split())
        forbidden[c - 1].add(d - 1)              # convert to 0-based indices

    # trivial case : no restriction
    if L == 0:
        print(max(a) + max(b))
        return

    # side dishes sorted by price (index list, descending by price)
    side_order = sorted(range(M), key=lambda idx: -b[idx])

    best = 0
    for i in range(N):
        bad = forbidden[i]
        # scan the side dishes until we find one that is not forbidden
        for j in side_order:
            if j not in bad:
                best = max(best, a[i] + b[j])
                break                    # next main dish

    print(best)


if __name__ == "__main__":
    main()