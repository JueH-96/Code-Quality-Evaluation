k, g, m = map(int, input().split())

glass_water = 0
mug_water = 0

for _ in range(k):
    if glass_water == g:
        glass_water = 0
    elif mug_water == 0:
        mug_water = m
    else:
        transfer_amount = min(mug_water, g - glass_water)
        glass_water += transfer_amount
        mug_water -= transfer_amount

print(glass_water, mug_water)