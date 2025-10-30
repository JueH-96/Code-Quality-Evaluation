# YOUR CODE HERE
import sys

def solve():
    N, M, D = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    max_sum = -1
    i, j = N-1, M-1
    while i >= 0 and j >= 0:
        if A[i] + B[j] <= D:
            max_sum = A[i] + B[j]
            break
        elif A[i] > B[j]:
            i -= 1
        else:
            j -= 1

    print(max_sum)

solve()