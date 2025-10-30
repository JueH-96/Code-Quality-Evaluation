# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

max_count = 0
current_count = 0

for i in range(2**100 - A[-1] - 1):
    if i % 2 == 0:
        current_count += 1
    else:
        current_count -= 1

    if i + A[-1] in A:
        current_count += 1

    max_count = max(max_count, current_count)

print(max_count)