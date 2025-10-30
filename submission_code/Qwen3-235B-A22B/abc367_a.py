# Read the input values
A, B, C = map(int, input().split())

# Determine if A is in the sleeping interval
if B < C:
    if B <= A < C:
        print("No")
    else:
        print("Yes")
else:
    if A >= B or A < C:
        print("No")
    else:
        print("Yes")