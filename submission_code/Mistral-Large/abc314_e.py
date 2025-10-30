import numpy as np
from scipy.linalg import solve

def expected_cost(N, M, wheels):
    # Define the transition matrix and cost vector
    size = M * N + 1
    A = np.zeros((size, size))
    b = np.zeros(size)

    for points in range(M):
        for i in range(N):
            idx = points * N + i
            cost, probs, scores = wheels[i]
            A[idx, idx] = 1
            A[idx, M * N] = -1
            b[idx] = cost
            for score, prob in zip(scores, probs):
                if points + score < M:
                    A[idx, (points + score) * N:(points + score + 1) * N] = [-p / len(scores) for p in probs]

    A[M * N, M * N] = 1
    b[M * N] = 0

    # Solve the linear system
    x = solve(A, b)
    return x[M * N]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
wheels = []
index = 2

for i in range(N):
    C = int(data[index])
    P = int(data[index + 1])
    S = list(map(int, data[index + 2:index + 2 + P]))
    wheels.append((C, [1/P] * P, S))
    index += 2 + P

# Calculate the expected cost
result = expected_cost(N, M, wheels)
print(result)