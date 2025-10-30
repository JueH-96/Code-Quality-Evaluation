# YOUR CODE HERE
K, G, M = map(int, input().split())

glass = 0
mug = 0

for _ in range(K):
    if glass == G:
        # Glass is full, discard all water
        glass = 0
    elif mug == 0:
        # Mug is empty, fill it
        mug = M
    else:
        # Transfer from mug to glass
        transfer = min(G - glass, mug)
        glass += transfer
        mug -= transfer

print(glass, mug)