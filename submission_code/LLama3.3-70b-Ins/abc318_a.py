import math

def count_full_moons(n, m, p):
    """
    Calculate the number of full moons between day 1 and day N.

    Args:
    n (int): The last day to consider.
    m (int): The first day on which a full moon can be seen.
    p (int): The interval between full moons.

    Returns:
    int: The number of full moons between day 1 and day N.
    """
    # Calculate the number of full moons
    # The first full moon is on day m, so we subtract m from n and divide by p
    # We use math.floor to round down to the nearest whole number, because we can't have a fraction of a full moon
    # We add 1 because we're counting the first full moon as well
    count = math.floor((n - m) / p) + 1 if m <= n else 0
    
    return count

# Read input from stdin
n, m, p = map(int, input().split())

# Calculate and print the result
result = count_full_moons(n, m, p)
print(result)