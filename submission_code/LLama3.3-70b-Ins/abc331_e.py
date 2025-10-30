import sys

def solve():
    N, M, L = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    c = []
    d = []
    for _ in range(L):
        ci, di = map(int, sys.stdin.readline().split())
        c.append(ci - 1)
        d.append(di - 1)

    max_price = 0
    for i in range(N):
        for j in range(M):
            if (i, j) not in list(zip(c, d)):
                max_price = max(max_price, a[i] + b[j])

    print(max_price)

if __name__ == "__main__":
    solve()