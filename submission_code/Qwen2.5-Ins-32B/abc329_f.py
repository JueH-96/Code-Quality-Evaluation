import sys
from collections import defaultdict

input = sys.stdin.read
sys.setrecursionlimit(2 * 10**5 + 100)

N, Q, *queries = map(int, input().split())
C = queries[:N]
queries = queries[N:]

boxes = defaultdict(set)
for i, c in enumerate(C, 1):
    boxes[i].add(c)

results = []
for a, b in zip(queries[::2], queries[1::2]):
    if a != b:
        boxes[b].update(boxes[a])
        boxes[a].clear()
        results.append(len(boxes[b]))

sys.stdout.write('
'.join(map(str, results)) + '
')