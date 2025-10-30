K, G, M = map(int, input().split())
glass = 0
mug = 0

for _ in range(K):
    if glass == G:
        glass = 0
    else:
        if mug == 0:
            mug = M
        else:
            transfer = min(G - glass, mug)
            glass += transfer
            mug -= transfer
            if glass == G:
                glass = 0

print(glass, mug)