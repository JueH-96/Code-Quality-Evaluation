N, X, Y, Z = map(int, input().split())

if X < Y:
    # Inbound train: X to Y via increasing station numbers
    if X < Z < Y:
        print("Yes")
    else:
        print("No")
else:
    # Outbound train: X to Y via decreasing station numbers
    if X > Z > Y:
        print("Yes")
    else:
        print("No")