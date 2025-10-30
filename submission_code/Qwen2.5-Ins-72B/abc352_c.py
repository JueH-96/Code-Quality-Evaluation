# YOUR CODE HERE
import sys

def max_giant_height(N, giants):
    # Calculate the difference between head height and shoulder height for each giant
    differences = [B - A for A, B in giants]
    # Calculate the cumulative shoulder height
    cumulative_shoulder_height = sum(A for A, B in giants)
    # The maximum possible height of the head of the topmost giant
    max_height = cumulative_shoulder_height + max(differences)
    return max_height

# Read input
N = int(input())
giants = [tuple(map(int, input().split())) for _ in range(N)]

# Compute and print the result
print(max_giant_height(N, giants))