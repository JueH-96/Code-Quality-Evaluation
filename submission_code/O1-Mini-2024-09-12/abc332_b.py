# YOUR CODE HERE
K, G, M = map(int, input().split())
glass, mug = 0, 0
for _ in range(K):
    if glass == G:
        glass = 0
    elif mug == 0:
        mug = M
    else:
        transfer = min(mug, G - glass)
        glass += transfer
        mug -= transfer
print(glass, mug)