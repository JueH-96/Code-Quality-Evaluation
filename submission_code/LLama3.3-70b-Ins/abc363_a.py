def min_increase_in_rating(R):
    """
    Calculate the minimum increase in rating required to increase the number of displayed ^.

    Args:
    R (int): The current rating of Takahashi.

    Returns:
    int: The minimum increase in rating required.
    """
    # Define the ranges for each number of displayed ^
    ranges = [(1, 99), (100, 199), (200, 299), (300, 399)]

    # Find the current range of the rating
    for i, (low, high) in enumerate(ranges):
        if low <= R <= high:
            # If the rating is already at the maximum of its range, move to the next range
            if R == high:
                return 0
            # Calculate the minimum increase in rating required to reach the next range
            next_low = ranges[i + 1][0] if i < len(ranges) - 1 else float('inf')
            return next_low - R

# Read the input from stdin
R = int(input())

# Calculate and print the minimum increase in rating required
print(min_increase_in_rating(R))