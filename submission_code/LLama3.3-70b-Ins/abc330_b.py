import sys

def solve():
    N, L, R = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    X = []
    for i in range(N):
        if A[i] < L:
            X.append(L)
        elif A[i] > R:
            X.append(R)
        else:
            X.append(A[i])

    print(*X)

if __name__ == "__main__":
    solve()