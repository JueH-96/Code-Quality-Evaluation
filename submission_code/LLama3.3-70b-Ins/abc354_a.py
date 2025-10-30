def find_day(H):
    """
    Find the first day such that the plant's height is greater than Takahashi's height in the morning.

    Args:
    H (int): Takahashi's height.

    Returns:
    int: The first day such that the plant's height is greater than Takahashi's height in the morning.
    """
    plant_height = 0
    day = 0

    # Continue to grow the plant until its height is greater than Takahashi's height
    while plant_height <= H:
        day += 1
        # The plant grows by 2^i cm on day i's night
        plant_height += 2 ** (day - 1)

    return day

# Read the input from stdin
H = int(input())

# Find and print the first day such that the plant's height is greater than Takahashi's height
print(find_day(H))