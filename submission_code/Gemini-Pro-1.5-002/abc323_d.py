# YOUR CODE HERE
from collections import defaultdict

n = int(input())
slimes = defaultdict(int)
for _ in range(n):
    s, c = map(int, input().split())
    slimes[s] = c

total_slimes = sum(slimes.values())
for size in sorted(slimes.keys()):
    count = slimes[size]
    synthesized = count // 2
    total_slimes -= synthesized
    slimes[size * 2] += synthesized
    slimes[size] %= 2
    if slimes[size] == 0:
        del slimes[size]

print(sum(1 for count in slimes.values() if count > 0))