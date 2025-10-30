# YOUR CODE HERE

# Read the inputs
N, Q = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the number of teeth
teeth = N

# Perform the treatments
for i in range(Q):
    if T[i] in range(1, N+1):
        teeth -= 1
    else:
        teeth += 1

# Print the number of teeth
print(teeth)