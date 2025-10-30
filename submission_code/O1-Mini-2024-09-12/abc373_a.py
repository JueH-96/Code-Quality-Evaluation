s = [input().strip() for _ in range(12)]
count = sum(1 for i, si in enumerate(s, 1) if len(si) == i)
print(count)