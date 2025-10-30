N, X, Y, Z = map(int, input().split())

# If X < Y, must use inbound train (going from smaller to larger station numbers)
# If X > Y, must use outbound train (going from larger to smaller station numbers)

if X < Y:  # Using inbound train
    # For inbound train, Z must be between X and Y and greater than X
    if X < Z < Y:
        print("Yes")
    else:
        print("No")
else:  # Using outbound train
    # For outbound train, Z must be between Y and X and less than X
    if Y < Z < X:
        print("Yes")
    else:
        print("No")