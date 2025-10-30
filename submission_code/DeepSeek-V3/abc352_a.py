# YOUR CODE HERE
N, X, Y, Z = map(int, input().split())

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