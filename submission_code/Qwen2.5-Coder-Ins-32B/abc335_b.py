# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())

for x in range(N + 1):
    for y in range(N - x + 1):
        for z in range(N - x - y + 1):
            print(x, y, z)