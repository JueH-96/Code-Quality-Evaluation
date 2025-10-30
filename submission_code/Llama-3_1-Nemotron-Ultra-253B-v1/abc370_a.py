# Read the input values
L, R = map(int, input().split())

# Calculate the sum of the hands raised
total = L + R

# Determine the output based on the sum and individual values
if total == 1:
    print("Yes" if L == 1 else "No")
else:
    print("Invalid")