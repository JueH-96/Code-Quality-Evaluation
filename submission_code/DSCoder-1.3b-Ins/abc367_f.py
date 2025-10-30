import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    for _ in range(Q):
        l, r, L, R = map(int, sys.stdin.readline().split())
        if all(A[i] == B[i] for i in range(l, r + 1)) and all(A[i] == B[i] for i in range(L, R + 1)):
            print('Yes')
        else:
            print('No')

if __name__ == "__main__":
    solve()