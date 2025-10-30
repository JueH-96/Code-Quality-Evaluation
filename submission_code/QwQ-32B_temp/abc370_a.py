# Read the input values
L, R = map(int, input().split())

# Check the sum of L and R
total = L + R

if total != 1:
    print("Invalid")
else:
    if L == 1:
        print("Yes")
    else:
        print("No")