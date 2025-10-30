import math

def min_value(D):
    """
    Find the minimum value of |x^2+y^2-D| for non-negative integers x and y.
    
    Args:
    D (int): A positive integer.
    
    Returns:
    int: The minimum value of |x^2+y^2-D|.
    """
    
    # Initialize the minimum difference
    min_diff = float('inf')
    
    # Iterate over all possible values of x
    for x in range(int(math.sqrt(D)) + 1):
        # Calculate the corresponding y value
        y = math.sqrt(D - x**2)
        
        # If y is an integer, update the minimum difference
        if y == int(y):
            min_diff = 0
            break
        
        # Calculate the difference for the floor and ceiling of y
        y_floor = math.floor(y)
        y_ceil = math.ceil(y)
        
        # Update the minimum difference
        min_diff = min(min_diff, abs(x**2 + y_floor**2 - D), abs(x**2 + y_ceil**2 - D))
    
    return min_diff

# Read the input from stdin
D = int(input())

# Print the answer to stdout
print(min_value(D))