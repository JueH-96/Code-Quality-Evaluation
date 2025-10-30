# Read the input values
L, R = map(int, input().split())

# Check if the sum of L and R is not 1 (either 0 or 2)
if L + R != 1:
    print("Invalid")
else:
    # Check which hand is raised and print accordingly
    if L == 1:
        print("Yes")
    else:
        print("No")