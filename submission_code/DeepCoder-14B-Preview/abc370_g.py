# Read the input values for L and R
L, R = map(int, input().split())

# Calculate the sum of L and R
total = L + R

# Check the conditions and print the appropriate result
if total == 1:
    if L == 1:
        print("Yes")
    else:
        print("No")
else:
    print("Invalid")