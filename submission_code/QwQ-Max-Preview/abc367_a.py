# Read the input values
A, B, C = map(int, input().split())

# Check if B is less than C
if B < C:
    # Check if A is in the sleeping interval [B, C)
    if B <= A < C:
        print("No")
    else:
        print("Yes")
else:
    # Check if A is in the sleeping intervals [B, 24) or [0, C)
    if A >= B or A < C:
        print("No")
    else:
        print("Yes")