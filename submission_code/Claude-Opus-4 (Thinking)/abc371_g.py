# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Keep track of all configurations
configurations = []
current = A.copy()
configurations.append(current.copy())

while True:
    # Apply the permutation
    new_current = [0] * N
    for i in range(N):
        new_current[i] = current[P[i] - 1]  # P[i] is 1-indexed, so subtract 1
    current = new_current
    
    if current == A:
        break
    configurations.append(current.copy())

# Find the lexicographically smallest
min_config = min(configurations)
print(*min_config)