import math
import sys

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def min_distance_with_penalty(N, checkpoints):
    dp = [float('inf')] * (1 << N)
    dp[1] = 0  # Start from checkpoint 1

    for mask in range(1, 1 << N):
        if mask & 1 == 0:  # Checkpoint 1 must be included
            continue
        for i in range(1, N):
            if mask & (1 << i) == 0:  # Checkpoint i is not in the current mask
                continue
            for j in range(1, N):
                if i == j or mask & (1 << j) == 0:  # Checkpoint j is not in the current mask or is the same as i
                    continue
                new_mask = mask ^ (1 << i)
                distance = euclidean_distance(checkpoints[j][0], checkpoints[j][1], checkpoints[i][0], checkpoints[i][1])
                penalty = 2 ** (bin(mask).count('1') - 2) if bin(mask).count('1') > 1 else 0
                dp[mask] = min(dp[mask], dp[new_mask] + distance + penalty)

    return dp[(1 << N) - 1]

# Read input
N = int(input())
checkpoints = [tuple(map(int, input().split())) for _ in range(N)]

# Add a dummy checkpoint at the end to handle the last transition
checkpoints.append((checkpoints[-1][0], checkpoints[-1][1]))

# Calculate the minimum distance with penalty
result = min_distance_with_penalty(N, checkpoints)

# Print the result
print(result)