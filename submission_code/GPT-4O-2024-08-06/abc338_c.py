# YOUR CODE HERE
def max_servings(N, Q, A, B):
    max_servings = 0
    # Try all possible combinations of servings of dish A (x) and dish B (y)
    for x in range(0, 1000001):  # Arbitrarily large number, will break early
        for y in range(0, 1000001):
            valid = True
            for i in range(N):
                if x * A[i] + y * B[i] > Q[i]:
                    valid = False
                    break
            if valid:
                max_servings = max(max_servings, x + y)
            else:
                break  # No need to increase y further if this y is not valid
        if x * min(A) > max(Q):  # If even the smallest A_i * x exceeds max Q_i, break
            break
    return max_servings

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))
B = list(map(int, data[2*N+1:3*N+1]))

print(max_servings(N, Q, A, B))