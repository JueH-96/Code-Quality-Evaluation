# YOUR CODE HERE

# Read the inputs
N, X, Y, Z = map(int, input().split())

# Calculate the distance from X to Y
distance = abs(Y - X)

# If the distance is divisible by 2, the train stops at Z
if distance % 2 == 0:
    print("Yes")
else:
    print("No")