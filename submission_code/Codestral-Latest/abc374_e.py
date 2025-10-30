import sys
from itertools import product

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
A = [int(data[2 * i + 2]) for i in range(N)]
P = [int(data[2 * i + 3]) for i in range(N)]
B = [int(data[2 * i + 4]) for i in range(N)]
Q = [int(data[2 * i + 5]) for i in range(N)]

def max_production_capacity(N, X, A, P, B, Q):
    max_capacity = 0

    for num_machines in product(range(X // min(P) + 1), repeat=N):
        total_cost = 0
        capacities = []

        for i in range(N):
            num_S = num_machines[i]
            num_T = (X - total_cost) // Q[i] if Q[i] > 0 else 0
            total_cost += num_S * P[i] + num_T * Q[i]
            capacities.append(A[i] * num_S + B[i] * num_T)

        if total_cost <= X:
            max_capacity = max(max_capacity, min(capacities))

    return max_capacity

result = max_production_capacity(N, X, A, P, B, Q)
print(result)