# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
heights = list(map(int, input[1:]))

first_height = heights[0]
for i in range(1, N):
    if heights[i] > first_height:
        print(i + 1)
        break
else:
    print(-1)