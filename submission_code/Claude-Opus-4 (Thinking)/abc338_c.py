# YOUR CODE HERE
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

max_servings = 0

# Find the maximum number of servings of dish A we can make
max_a = float('inf')
for i in range(N):
    if A[i] > 0:
        max_a = min(max_a, Q[i] // A[i])

if max_a == float('inf'):
    max_a = 0
else:
    max_a = int(max_a)

# Try all possible numbers of servings of dish A
for a in range(max_a + 1):
    # Calculate the maximum number of servings of dish B we can make
    max_b = float('inf')
    for i in range(N):
        remaining = Q[i] - a * A[i]
        if B[i] > 0:
            max_b = min(max_b, remaining // B[i])
    
    if max_b == float('inf'):
        max_b = 0
    else:
        max_b = int(max_b)
    
    total = a + max_b
    max_servings = max(max_servings, total)

print(max_servings)