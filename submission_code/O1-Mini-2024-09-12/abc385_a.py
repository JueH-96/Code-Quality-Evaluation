# Read three integers
A, B, C = map(int, input().split())

# Check if any one equals the sum of the other two, or all three are equal
if A == B + C or B == A + C or C == A + B or (A == B and B == C):
    print("Yes")
else:
    print("No")