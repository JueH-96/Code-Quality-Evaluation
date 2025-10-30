def water_in_glass_and_mug(K, G, M):
    glass = 0
    mug = 0

    for _ in range(K):
        if glass == G:
            glass = 0
        elif mug == 0:
            mug = M
        else:
            transfer_amount = min(G - glass, mug)
            glass += transfer_amount
            mug -= transfer_amount

    return glass, mug

# Read input
import sys
input = sys.stdin.read
K, G, M = map(int, input().strip().split())

# Get the result
result = water_in_glass_and_mug(K, G, M)

# Print the result
print(result[0], result[1])