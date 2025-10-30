k, g, m = map(int, input().split())

glass = 0
mug = 0

for _ in range(k):
    if glass == g:
        glass = 0
    elif mug == 0:
        mug = m
    else:
        transfer = min(m - mug, g - glass)
        glass += transfer
        mug -= transfer

print(glass, mug)