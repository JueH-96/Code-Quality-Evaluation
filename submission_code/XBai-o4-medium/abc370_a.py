# Read the input values
L, R = map(int, input().split())

# Check the sum of L and R to determine the scenario
total = L + R

if total != 1:
    print("Invalid")
else:
    # If L is 1, output Yes; otherwise, output No
    print("Yes" if L == 1 else "No")