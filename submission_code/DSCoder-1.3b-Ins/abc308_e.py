import sys

def mex(s):
    i = 0
    while i in s:
        i += 1
    return i

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    S = sys.stdin.readline().strip()

    mex_A = [mex(A[i] for i in range(N)) for _ in range(N)]
    mex_S = [mex(S[i] for i in range(N))]

    result = sum(mex_A[i] != mex_S[0] for i in range(N))

    print(result)

solve()