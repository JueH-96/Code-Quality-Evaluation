import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]

    for query in queries:
        if query[0] == 1:
            i, x = query[1:]
            A[i-1] = x
        elif query[0] == 2:
            i, x = query[1:]
            B[i-1] = x
        else:
            l, r = query[1:]
            v = 0
            for i in range(l-1, r):
                if v == 0:
                    v = A[i]
                else:
                    v = max(v + A[i], v * B[i])
            print(v)

solve()