# Read input
N, X, Y, Z = map(int, input().split())

# Determine which type of train Takahashi will use based on station order
if X < Y:  # Inbound train (from station 1 to station N)
    # Check if Z is in the path between X and Y
    if X < Z < Y:
        print("Yes")
    else:
        print("No")
else:  # Outbound train (from station N to station 1)
    # Check if Z is in the path between X and Y
    if X > Z > Y:
        print("Yes")
    else:
        print("No")