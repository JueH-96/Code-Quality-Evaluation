import sys
data = sys.stdin.read().split()
N, X, Y, Z = map(int, data)

if X < Y:
    if X < Z <= Y:
        print("Yes")
    else:
        print("No")
elif X > Y:
    if Y <= Z < X:
        print("Yes")
    else:
        print("No")