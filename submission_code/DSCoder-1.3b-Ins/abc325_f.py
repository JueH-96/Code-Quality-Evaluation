# YOUR CODE HERE

import sys

def solve():
    N = int(sys.stdin.readline().strip())
    D = list(map(int, sys.stdin.readline().strip().split()))
    L = list(map(int, sys.stdin.readline().strip().split()))
    C = list(map(int, sys.stdin.readline().strip().split()))
    K = list(map(int, sys.stdin.readline().strip().split()))

    if sum(D) < N:
        print(-1)
        return

    D.sort()
    C.sort()

    total_cost = 0
    for i in range(N):
        if D[i] > L[i]:
            print(-1)
            return
        total_cost += C[i]
        K[i] -= 1
        if K[i] == 0:
            total_cost += C[i]

    print(total_cost)

solve()