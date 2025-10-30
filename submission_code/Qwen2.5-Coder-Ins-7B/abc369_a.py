# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])

count = 0

for x in range(-100, 101):
    if (A - x) * (B - x) == (A - B) ** 2:
        count += 1

print(count)