import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if A[j] - A[i] == A[1] - A[0] and A[j] - A[i] == A[2] - A[1]:
                count += 1

    print(count)

solve()