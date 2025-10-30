from collections import defaultdict

n = int(input())
beans = defaultdict(list)

for _ in range(n):
    deliciousness, color = map(int, input().split())
    beans[color].append(deliciousness)

max_min_deliciousness = float('-inf')

for color in beans:
    min_deliciousness = min(beans[color])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)