# YOUR CODE HERE
import sys

input = sys.stdin.read().splitlines()
N = int(input[0])
count = 0

for i in range(1, N + 1):
    if input[i] == "Takahashi":
        count += 1

print(count)