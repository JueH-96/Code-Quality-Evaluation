K, G, M = map(int, input().split())
g = 0
m = 0

for _ in range(K):
    if g == G:
        g = 0
    else:
        if m == 0:
            m = M
        else:
            transfer = min(G - g, m)
            g += transfer
            m -= transfer

print(g, m)