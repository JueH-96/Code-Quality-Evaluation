# YOUR CODE HERE
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

min_servings = float('inf')

for i in range(N):
    if A[i] > 0:
        min_servings = min(min_servings, Q[i] // A[i])
    if B[i] > 0:
        min_servings = min(min_servings, Q[i] // B[i])

print(min_servings)