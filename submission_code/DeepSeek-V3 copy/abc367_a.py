# Read input values
A, B, C = map(int, input().split())

# Determine if Takahashi is awake at A o'clock
if B < C:
    if B <= A < C:
        print("Yes")
    else:
        print("No")
else:
    if A < C or A >= B:
        print("Yes")
    else:
        print("No")