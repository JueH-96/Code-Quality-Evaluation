# YOUR CODE HERE
from collections import defaultdict

a, b, c, d = map(int, input().split())

count = defaultdict(int)
for num in [a, b, c, d]:
    count[num] += 1

counts = sorted(count.values(), reverse=True)

if (len(counts) >= 2 and counts[0] == 3 and counts[1] == 1) or (len(counts) >= 2 and counts[0] == 2 and counts[1] == 2):
    print("Yes")
else:
    print("No")