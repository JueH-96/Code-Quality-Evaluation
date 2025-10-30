# Read input values
A, B, C = map(int, input().split())

# Determine if Takahashi is awake at time A
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