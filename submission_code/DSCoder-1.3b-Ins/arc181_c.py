import sys

def solve():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))
    Q = list(map(int, sys.stdin.readline().strip().split()))

    S = [''.join(map(str, sorted([2**i for i in range(N)] + [1]*N)))]
    for p in P:
        S.append(''.join(map(str, sorted([2**i for i in range(N)] + [1]*p))))
    for q in Q:
        S.append(''.join(map(str, sorted([2**i for i in range(N)] + [1]*q))))

    T = [''.join(map(str, sorted([2**i for i in range(N)] + [1]*N, reverse=True)))]
    for p in P:
        T.append(''.join(map(str, sorted([2**i for i in range(N)] + [1]*p, reverse=True))))
    for q in Q:
        T.append(''.join(map(str, sorted([2**i for i in range(N)] + [1]*q, reverse=True))))

    A = [['0']*N for _ in range(N)]
    for i in range(N):
        A[i][P[i]-1] = '1'
        A[i][Q[i]-1] = '1'

    for i in range(N):
        print(''.join(A[i]))

solve()