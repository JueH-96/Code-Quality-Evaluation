# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

X = int(data[0])
Y = int(data[1])

if abs(X - Y) <= 2:
    print("Yes")
else:
    print("No")