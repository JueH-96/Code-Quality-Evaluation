# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
A = list(map(int, input[2:]))

empty_seats = K
starts = 0

for group in A:
    if group > empty_seats:
        starts += 1
        empty_seats = K - group
    else:
        empty_seats -= group
    if empty_seats == 0:
        starts += 1
        empty_seats = K

# If there are any remaining people that haven't started the attraction yet
if empty_seats < K:
    starts += 1

print(starts)