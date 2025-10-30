import sys

def solve():
    T = int(sys.stdin.readline().strip())
    for t in range(T):
        N = int(sys.stdin.readline().strip())
        A = list(map(int, sys.stdin.readline().strip().split()))
        B = list(map(int, sys.stdin.readline().strip().split()))
        C = list(map(int, sys.stdin.readline().strip().split()))

        count = 0
        for i in range(N):
            if A[i] * B[i] < C[i]:
                count += 1

        print(count)

solve()