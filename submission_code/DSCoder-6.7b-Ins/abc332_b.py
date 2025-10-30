# YOUR CODE HERE

K, G, M = map(int, input().split())

glass = 0
mug = M

for _ in range(K):
    if glass == 0:
        glass = G
        mug -= G
    else:
        if mug == M:
            mug = 0
            glass -= G
        else:
            diff = G - glass
            if diff > mug:
                glass += mug
                mug = 0
            else:
                glass = G
                mug -= diff

print(glass, mug)