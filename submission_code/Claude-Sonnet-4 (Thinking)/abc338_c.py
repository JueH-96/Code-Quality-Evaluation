N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

max_servings = 0

# Calculate maximum possible servings of dish A alone
max_A = float('inf')
for i in range(N):
    if A[i] > 0:
        max_A = min(max_A, Q[i] // A[i])

if max_A == float('inf'):
    max_A = 0

# Try all possible values of x (servings of dish A)
for x in range(max_A + 1):
    # For this x, find the maximum y (servings of dish B)
    max_y = float('inf')
    feasible = True
    
    for i in range(N):
        remaining = Q[i] - x * A[i]
        if remaining < 0:
            feasible = False
            break
        if B[i] > 0:
            max_y = min(max_y, remaining // B[i])
    
    if feasible:
        max_servings = max(max_servings, x + max_y)

print(max_servings)