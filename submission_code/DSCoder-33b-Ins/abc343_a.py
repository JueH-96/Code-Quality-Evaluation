# YOUR CODE HERE
import sys

A, B = map(int, sys.stdin.readline().split())

for i in range(10):
    if i != A + B:
        print(i)
        break