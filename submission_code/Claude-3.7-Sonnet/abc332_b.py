def water_operations(K, G, M):
    glass = 0  # Initially, the glass is empty
    mug = 0    # Initially, the mug is empty
    
    for _ in range(K):
        if glass == G:  # If the glass is filled to capacity
            glass = 0   # Discard all water from the glass
        elif mug == 0:  # If the mug is empty
            mug = M     # Fill the mug to its capacity
        else:           # Transfer water from the mug to the glass
            # Amount of water that can be transferred
            transfer_amount = min(G - glass, mug)
            glass += transfer_amount
            mug -= transfer_amount
    
    return glass, mug

# Read input
K, G, M = map(int, input().split())

# Perform operations and print the result
glass, mug = water_operations(K, G, M)
print(f"{glass} {mug}")