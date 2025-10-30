M, *rest = map(int, open(0).read().split())
D = rest[:M]
total = sum(D)
target = (total + 1) // 2
cumulative = 0
for i, days in enumerate(D, 1):
    if cumulative + days >= target:
        print(i, target - cumulative)
        break
    cumulative += days