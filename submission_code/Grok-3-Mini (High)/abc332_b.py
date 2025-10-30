# YOUR CODE HERE
K, G, M = map(int, input().split())
glass_water = 0
mug_water = 0
for _ in range(K):
    if glass_water == G:
        glass_water = 0
    else:
        if mug_water == 0:
            mug_water = M
        else:
            transfer_amount = min(mug_water, G - glass_water)
            glass_water += transfer_amount
            mug_water -= transfer_amount
print(glass_water, mug_water)