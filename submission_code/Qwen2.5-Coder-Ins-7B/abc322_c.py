# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Create a list to store the result
result = [0] * N

# Initialize the last day fireworks are launched
last_firework_day = N

# Iterate from the last day to the first day
for i in range(N-1, -1, -1):
    if i == last_firework_day:
        continue
    if i in A:
        last_firework_day = i
    result[i] = last_firework_day - i

# Print the result
for r in result:
    print(r)