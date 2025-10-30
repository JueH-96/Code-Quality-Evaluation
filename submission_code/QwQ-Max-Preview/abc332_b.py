k, G, M = map(int, input().split())
g, m = 0, 0

for _ in range(k):
    if g == G:
        g = 0
    else:
        if m == 0:
            m = M
        else:
            transfer = min(m, G - g)
            g += transfer
            m -= transfer

print(g, m)