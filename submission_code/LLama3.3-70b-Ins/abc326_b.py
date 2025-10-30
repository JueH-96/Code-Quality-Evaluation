def find_326_like_number(n):
    """
    Find the smallest 326-like number greater than or equal to n.

    Args:
    n (int): The input number.

    Returns:
    int: The smallest 326-like number greater than or equal to n.
    """
    while True:
        # Convert the number to a string to easily access each digit
        str_n = str(n)
        
        # Check if the number is a 326-like number
        if int(str_n[0]) * int(str_n[1]) == int(str_n[2]):
            return n
        
        # If not, increment the number and check again
        n += 1

# Read the input from stdin
n = int(input())

# Find and print the smallest 326-like number greater than or equal to n
print(find_326_like_number(n))