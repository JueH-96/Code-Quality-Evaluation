# YOUR CODE HERE
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

max_servings = 0

# Find maximum possible servings of dish A
max_a = float('inf')
for i in range(N):
    if A[i] > 0:
        max_a = min(max_a, Q[i] // A[i])

if max_a == float('inf'):
    max_a = 0

# Try each possible number of dish A servings
for x in range(max_a + 1):
    # Calculate remaining ingredients after making x servings of dish A
    remaining = [Q[i] - x * A[i] for i in range(N)]
    
    # Find maximum servings of dish B with remaining ingredients
    max_b = float('inf')
    for i in range(N):
        if B[i] > 0:
            max_b = min(max_b, remaining[i] // B[i])
    
    if max_b == float('inf'):
        max_b = 0
    
    # Update maximum total servings
    max_servings = max(max_servings, x + max_b)

print(max_servings)