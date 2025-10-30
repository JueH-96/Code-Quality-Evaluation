from collections import Counter

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()
    counter = Counter(s)
    odd_count = sum(v % 2 for v in counter.values())
    if odd_count <= k and (k - odd_count) % 2 == 0:
        print('YES')
    else:
        print('NO')