# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

X = int(data[0])
Y = int(data[1])

if X < Y:
    if Y - X <= 2:
        print("Yes")
    else:
        print("No")
else:
    if X - Y <= 3:
        print("Yes")
    else:
        print("No")