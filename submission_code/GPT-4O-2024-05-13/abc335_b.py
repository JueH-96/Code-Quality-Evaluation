# YOUR CODE HERE
import sys
input = sys.stdin.read

N = int(input().strip())

for x in range(N + 1):
    for y in range(N + 1):
        for z in range(N + 1):
            if x + y + z <= N:
                print(x, y, z)