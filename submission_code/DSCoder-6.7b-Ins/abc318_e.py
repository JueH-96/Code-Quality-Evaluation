from collections import defaultdict

N = int(input().strip())
A = list(map(int, input().strip().split()))

counts = defaultdict(int)
for a in A:
    counts[a] += 1

total = 0
for v in counts.values():
    total += v * (v-1) * (v-2) // 6

print(total)