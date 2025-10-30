# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

count = 0
seen = set()

for char in S:
    count += 1
    seen.add(char)
    if len(seen) == 3:
        break

print(count)