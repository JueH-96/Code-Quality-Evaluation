# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])
C = int(data[2])

if (A < B < C) or (C < A < B) or (B < C < A):
    print("Yes")
else:
    print("No")