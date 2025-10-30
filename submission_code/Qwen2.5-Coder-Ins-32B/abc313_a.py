# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
P = list(map(int, input[1:]))

max_other = max(P[1:])
needed_points = max(0, max_other - P[0] + 1)

print(needed_points)