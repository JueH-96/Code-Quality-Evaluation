N = int(input())
bases = []
for _ in range(N):
    W, X = map(int, input().split())
    bases.append((W, X))

max_total = 0

for T in range(24):
    total = 0
    for W, X in bases:
        L = (T + X) % 24
        if 9 <= L <= 17:
            total += W
    if total > max_total:
        max_total = total

print(max_total)