# YOUR CODE HERE
K, G, M = map(int, input().split())
glass_vol = 0
mug_vol = 0

for _ in range(K):
    if glass_vol == G:
        glass_vol = 0  # Discard all water from the glass
    else:
        if mug_vol == 0:
            mug_vol = M  # Fill the mug with water
        else:
            transfer_amount = min(mug_vol, G - glass_vol)
            glass_vol += transfer_amount
            mug_vol -= transfer_amount

print(f"{glass_vol} {mug_vol}")