k, g, m = map(int, input().split())
glass, mug = 0, 0
for _ in range(k):
    if glass == g:
        glass = 0
    elif not mug:
        mug = m
    else:
        glass += mug
        mug = 0
        if glass > g:
            mug = glass - g
            glass = g
print(glass, mug)