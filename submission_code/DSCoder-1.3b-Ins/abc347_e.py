import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = [0]*N
    S = set()

    for _ in range(Q):
        x = int(sys.stdin.readline())
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        for j in S:
            A[j-1] += len(S)
    print(' '.join(map(str, A)))

solve()