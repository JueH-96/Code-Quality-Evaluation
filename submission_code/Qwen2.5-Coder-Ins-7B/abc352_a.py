# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
Z = int(data[3])

if (X < Z < Y) or (Y < Z < X):
    print("Yes")
else:
    print("No")