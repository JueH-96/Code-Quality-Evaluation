def min_tshirts_to_buy(n, m, s):
    """
    Calculate the minimum number of T-shirts Takahashi needs to buy.

    Args:
    n (int): The number of days.
    m (int): The number of plain T-shirts.
    s (str): The schedule for N days.

    Returns:
    int: The minimum number of T-shirts Takahashi needs to buy.
    """
    plain_tshirts = m
    logo_tshirts = 0
    max_logo_tshirts_needed = 0

    for day in s:
        if day == '0':
            # Wash all T-shirts
            plain_tshirts = m
            logo_tshirts = 0
        elif day == '1':
            # Wear a plain or logo T-shirt
            if plain_tshirts > 0:
                plain_tshirts -= 1
            else:
                logo_tshirts += 1
        elif day == '2':
            # Wear a logo T-shirt
            logo_tshirts += 1

        # Update the maximum number of logo T-shirts needed
        max_logo_tshirts_needed = max(max_logo_tshirts_needed, logo_tshirts)

    return max_logo_tshirts_needed


# Read the inputs from stdin
n, m = map(int, input().split())
s = input()

# Calculate and print the minimum number of T-shirts to buy
print(min_tshirts_to_buy(n, m, s))