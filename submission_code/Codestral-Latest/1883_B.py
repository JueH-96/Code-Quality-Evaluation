import sys
from collections import Counter

input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1

results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    index += 3

    counter = Counter(s)
    odd_count = sum(1 for count in counter.values() if count % 2 != 0)

    if odd_count <= k + 1:
        results.append("YES")
    else:
        results.append("NO")

print("
".join(results))