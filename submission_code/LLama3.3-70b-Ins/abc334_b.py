import math

def count_christmas_trees(A, M, L, R):
    """
    Calculate the number of Christmas trees between Takahashi and Aoki.

    Args:
    A (int): The starting point of the Christmas trees.
    M (int): The interval between Christmas trees.
    L (int): The position of Takahashi.
    R (int): The position of Aoki.

    Returns:
    int: The number of Christmas trees between Takahashi and Aoki.
    """
    # Calculate the first Christmas tree that is not to the west of Takahashi
    first_tree = math.ceil((L - A) / M) * M + A
    
    # Calculate the last Christmas tree that is not to the east of Aoki
    last_tree = math.floor((R - A) / M) * M + A
    
    # Calculate the number of Christmas trees between Takahashi and Aoki
    num_trees = (last_tree - first_tree) // M + 1
    
    return num_trees

# Read input from stdin
A, M, L, R = map(int, input().split())

# Calculate and print the number of Christmas trees
print(count_christmas_trees(A, M, L, R))