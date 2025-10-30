# YOUR CODE HERE
import sys

A, B = map(int, sys.stdin.read().split())
sum_AB = A + B

# Find any number between 0 and 9 that is not equal to sum_AB
for num in range(10):
    if num != sum_AB:
        print(num)
        break