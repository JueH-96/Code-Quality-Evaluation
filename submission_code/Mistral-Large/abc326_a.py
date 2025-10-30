import sys

input = sys.stdin.read
data = input().split()

X = int(data[0])
Y = int(data[1])

if (Y > X and Y - X <= 2) or (X > Y and X - Y <= 3):
    print("Yes")
else:
    print("No")