# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the starting point of the cycle
start = 1
while True:
    if A[start-1] == start:
        start += 1
    else:
        break

# Find the cycle
cycle = []
current = start
while True:
    cycle.append(current)
    current = A[current-1]
    if current == start:
        break

# Print the result
print(len(cycle))
print(' '.join(map(str, cycle)))