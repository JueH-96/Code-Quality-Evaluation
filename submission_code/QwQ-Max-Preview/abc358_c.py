n, m = map(int, input().split())
s = [input().strip() for _ in range(n)]

min_stands = n  # Initialize with maximum possible

for mask in range(1, 1 << n):
    selected = []
    for i in range(n):
        if mask & (1 << i):
            selected.append(i)
    covered = set()
    for stand in selected:
        for j in range(m):
            if s[stand][j] == 'o':
                covered.add(j)
    if len(covered) == m:
        if len(selected) < min_stands:
            min_stands = len(selected)

print(min_stands)