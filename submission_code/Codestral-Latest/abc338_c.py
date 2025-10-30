import sys
from itertools import product

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))
B = list(map(int, data[2*N+1:3*N+1]))

max_servings = 0

# Iterate over all possible combinations of servings of dish A and dish B
for a, b in product(range((Q[0] // A[0]) + 1), range((Q[0] // B[0]) + 1)):
    can_make = True
    for i in range(1, N):
        if A[i] * a + B[i] * b > Q[i]:
            can_make = False
            break
    if can_make:
        max_servings = max(max_servings, a + b)

print(max_servings)