# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

count = 0
for i in range(1, N + 1):
    positions = [index for index, value in enumerate(A) if value == i]
    if len(positions) == 2 and abs(positions[0] - positions[1]) == 2:
        count += 1

print(count)