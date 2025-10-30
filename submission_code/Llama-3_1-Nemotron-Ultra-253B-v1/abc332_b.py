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
            transfer = min(mug, G - glass)
            mug -= transfer
            glass += transfer

print(glass, mug)