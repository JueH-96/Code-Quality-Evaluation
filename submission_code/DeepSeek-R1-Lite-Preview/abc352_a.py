# Read input values
N, X, Y, Z = map(int, input().split())

# Determine the direction of travel and check if Z is on the path
if X < Y:
    # Inbound travel: X < Z <= Y
    if X < Z <= Y:
        print("Yes")
    else:
        print("No")
elif X > Y:
    # Outbound travel: Y <= Z < X
    if Y <= Z < X:
        print("Yes")
    else:
        print("No")