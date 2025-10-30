# Read input
N, X, Y, Z = map(int, input().split())

# Check conditions for inbound and outbound trains
if (X < Y and X <= Z <= Y) or (X > Y and Y <= Z <= X):
    print("Yes")
else:
    print("No")