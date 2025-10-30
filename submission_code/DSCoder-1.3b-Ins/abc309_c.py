import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    a = []
    b = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        a.append(x)
        b.append(y)
    a.sort()
    b.sort()
    for i in range(N):
        if a[i] >= K:
            print(b[i])
            return
    print(a[-1])

solve()