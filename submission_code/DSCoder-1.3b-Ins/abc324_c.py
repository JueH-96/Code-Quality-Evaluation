import sys

def solve():
    N, T = map(str, sys.stdin.readline().strip().split())
    T = list(T)
    S = [str(sys.stdin.readline().strip()) for _ in range(int(N))]

    res = []
    for s in S:
        if len(s) != len(T):
            continue
        if s == T:
            res.append(1)
        elif s[0] == T[0] and s[-1] == T[-1]:
            res.append(2)
        elif s[0] == T[0] or s[-1] == T[-1]:
            res.append(3)
        elif len(set(zip(s, T))) == 1:
            res.append(4)

    print(len(res))
    print(*res)

solve()