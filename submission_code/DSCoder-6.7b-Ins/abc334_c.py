import sys

def min_weirdness():
    N, K = map(int, sys.stdin.readline().split())
    lost_colors = list(map(int, sys.stdin.readline().split()))

    # Sort the lost colors
    lost_colors.sort()

    # Initialize the total weirdness to 0
    total_weirdness = 0

    # Initialize the left and right pointers for the lost colors
    left = 0
    right = N - 1

    # Iterate over the remaining colors
    for i in range(1, N + 1):
        # If the current color is in the lost colors, move the right pointer
        if i == lost_colors[right]:
            right -= 1
        # If the current color is not in the lost colors, add the weirdness to the total
        else:
            total_weirdness += abs(i - lost_colors[left])
            left += 1

    return total_weirdness

print(min_weirdness())