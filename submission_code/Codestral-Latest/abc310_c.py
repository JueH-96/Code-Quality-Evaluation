# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:N+1]

unique_sticks = set()

for s in strings:
    if s not in unique_sticks and s[::-1] not in unique_sticks:
        unique_sticks.add(s)

print(len(unique_sticks))