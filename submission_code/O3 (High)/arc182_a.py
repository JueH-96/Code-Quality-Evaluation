import sys

MOD = 998244353

def main() -> None:
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    P = []
    V = []
    for _ in range(Q):
        p, v = map(int, input().split())
        P.append(p)
        V.append(v)

    # -1 : not decided, 0 : prefix, 1 : suffix
    kind = [-1] * Q

    for i in range(Q):
        pi, vi = P[i], V[i]
        for k in range(i + 1, Q):
            if vi <= V[k]:
                continue          # no constraint
            pk = P[k]
            if pi == pk:
                print(0)
                return
            # decide the only possible combination
            if pi < pk:
                # i -> prefix (0), k -> suffix (1)
                if kind[i] == 1 or kind[k] == 0:
                    print(0)
                    return
                kind[i] = 0
                kind[k] = 1
            else:  # pi > pk
                # i -> suffix (1), k -> prefix (0)
                if kind[i] == 0 or kind[k] == 1:
                    print(0)
                    return
                kind[i] = 1
                kind[k] = 0

    free = kind.count(-1)
    ans = pow(2, free, MOD)
    print(ans)

if __name__ == "__main__":
    main()