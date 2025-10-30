# YOUR CODE HERE
from bisect import bisect_right

def solve(N, A, B, C):
    pairs = 0
    for i in range(N):
        if A[i] == 0:
            y_max = (C[i] - 1) // B[i]
            pairs += y_max
        elif B[i] == 0:
            x_max = (C[i] - 1) // A[i]
            pairs += x_max
        else:
            x_max = (C[i] - 1) // A[i]
            y_max = (C[i] - 1) // B[i]
            if x_max * B[i] + y_max * A[i] >= C[i]:
                pairs += x_max * bisect_right(range(y_max + 1), (C[i] - A[i] * x_max - 1) // B[i])
            else:
                pairs += x_max * y_max
    return pairs

T = int(input())
for _ in range(T):
    N = int(input())
    AB = [list(map(int, input().split())) for _ in range(N)]
    print(solve(N, *zip(*AB)))