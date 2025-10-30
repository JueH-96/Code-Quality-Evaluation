import sys
from collections import defaultdict

input = sys.stdin.read
N, *AC = map(int, input().split())

beans = defaultdict(list)
for i in range(0, len(AC), 2):
    a, c = AC[i], AC[i + 1]
    beans[c].append(a)

min_deliciousness = 0
for color in beans:
    min_deliciousness = max(min_deliciousness, min(beans[color]))

print(min_deliciousness)