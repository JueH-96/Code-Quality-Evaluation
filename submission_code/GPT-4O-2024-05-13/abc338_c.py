# YOUR CODE HERE
def max_servings(N, Q, A, B):
    max_servings = 0
    for x in range(1000001):
        valid = True
        total_servings = x
        for i in range(N):
            if A[i] * x > Q[i]:
                valid = False
                break
        if not valid:
            break
        remaining_ingredients = [Q[i] - A[i] * x for i in range(N)]
        min_servings_B = float('inf')
        for i in range(N):
            if B[i] > 0:
                min_servings_B = min(min_servings_B, remaining_ingredients[i] // B[i])
        total_servings += min_servings_B
        max_servings = max(max_servings, total_servings)
    return max_servings

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
Q = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))
B = list(map(int, data[2*N+1:3*N+1]))

print(max_servings(N, Q, A, B))