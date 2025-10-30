# YOUR CODE HERE
import sys

input = sys.stdin.read().splitlines()

count = 0
for i in range(12):
    if len(input[i]) == i + 1:
        count += 1

print(count)