# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

count = 0
from collections import defaultdict
last_seen = defaultdict(list)

for k in range(N):
    for i in last_seen[A[k]]:
        if i < k - 1:
            count += 1
    last_seen[A[k]].append(k)

print(count)