# YOUR CODE HERE
K, G, M = map(int, input().split())

glass = 0
mug = 0

for _ in range(K):
    if glass == G:
        # Glass is full, discard all water from glass
        glass = 0
    elif mug == 0:
        # Mug is empty, fill the mug
        mug = M
    else:
        # Transfer water from mug to glass
        # Transfer until mug is empty or glass is full
        transfer_amount = min(mug, G - glass)
        glass += transfer_amount
        mug -= transfer_amount

print(glass, mug)