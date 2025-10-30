# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

empty_seats = K
attraction_starts = 0

for group_size in A:
    if empty_seats < group_size:
        attraction_starts += 1
        empty_seats = K
    empty_seats -= group_size

print(attraction_starts)