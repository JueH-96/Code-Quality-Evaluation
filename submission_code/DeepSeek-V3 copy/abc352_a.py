# Read input values
N, X, Y, Z = map(int, input().split())

# Determine the direction of travel
if X < Y:
    # Inbound train: X to Y
    if X <= Z <= Y:
        print("Yes")
    else:
        print("No")
else:
    # Outbound train: X to Y
    if Y <= Z <= X:
        print("Yes")
    else:
        print("No")