n = int(input())
a = list(map(int, input().split()))
from collections import defaultdict

indices = defaultdict(list)
for idx in range(len(a)):
    num = a[idx]
    indices[num].append(idx + 1)  # Convert to 1-based index

elements = []
for i in range(1, n + 1):
    positions = sorted(indices[i])
    mid = positions[1]
    elements.append((mid, i))

# Sort by the middle index, then by the integer value if needed
elements.sort(key=lambda x: (x[0], x[1]))

result = [str(x[1]) for x in elements]
print(' '.join(result))