# YOUR CODE HERE
K, G, M = map(int, input().split())
glass = 0
mug = M
for _ in range(K):
    if glass == G:
        glass = 0
    else:
        mug -= 1
        if mug == 0:
            mug = M
    glass += 1
print(glass, mug)