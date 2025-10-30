# Read the input from stdin
N, X, Y, Z = map(int, input().split())

# Determine if the train stops at station Z during the travel from X to Y
if X < Y:
    # Inbound train
    if X < Z < Y:
        print("Yes")
    else:
        print("No")
else:
    # Outbound train
    if Y < Z < X:
        print("Yes")
    else:
        print("No")