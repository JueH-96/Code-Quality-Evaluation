def find_tallest_building_position(building_heights):
    """
    This function finds the position of the building that is taller than the first building
    from the left. If no such building exists, it returns -1.

    Parameters:
    building_heights (list): A list of building heights.

    Returns:
    int: The position of the leftmost building that is taller than the first building, or -1 if none.
    """
    first_building_height = building_heights[0]
    for i, height in enumerate(building_heights[1:], start=1):
        if height > first_building_height:
            return i
    return -1

# Reading input from stdin and solving the problem
if __name__ == "__main__":
    N = int(input())  # Number of buildings
    building_heights = list(map(int, input().split()))  # Heights of the buildings
    result = find_tallest_building_position(building_heights)
    print(result)