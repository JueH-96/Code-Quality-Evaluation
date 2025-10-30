def can_eat_all_dishes(n, dishes):
    """
    Determine whether Takahashi can eat all the dishes without feeling sick.

    Args:
    n (int): The number of dishes.
    dishes (list): A list of strings representing the type of each dish.

    Returns:
    bool: True if Takahashi can eat all the dishes, False otherwise.
    """
    for i in range(n - 1):
        # Check if two consecutive dishes are sweet
        if dishes[i] == "sweet" and dishes[i + 1] == "sweet":
            # If they are, return False
            return False
    # If no two consecutive sweet dishes are found, return True
    return True


# Read the number of dishes from stdin
n = int(input())

# Read the type of each dish from stdin
dishes = [input() for _ in range(n)]

# Determine whether Takahashi can eat all the dishes
if can_eat_all_dishes(n, dishes):
    print("Yes")
else:
    print("No")