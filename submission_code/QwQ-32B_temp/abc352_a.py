# Read the input values
N, X, Y, Z = map(int, input().split())

# Determine if Z is on the path from X to Y
if (X < Y and (X <= Z <= Y)) or (X > Y and (Y <= Z <= X)):
    print("Yes")
else:
    print("No")