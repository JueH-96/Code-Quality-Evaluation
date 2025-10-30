# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
A = int(data[2])

if T > A + (N - T - A) / 2:
    print("Yes")
else:
    print("No")