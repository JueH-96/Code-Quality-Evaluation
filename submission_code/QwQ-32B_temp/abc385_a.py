# Read the input values
A, B, C = map(int, input().split())

# Calculate the total sum
total = A + B + C

# Check the two conditions
condition1 = (total % 2 == 0) and (A == total//2 or B == total//2 or C == total//2)
condition2 = (A == B == C)

if condition1 or condition2:
    print("Yes")
else:
    print("No")