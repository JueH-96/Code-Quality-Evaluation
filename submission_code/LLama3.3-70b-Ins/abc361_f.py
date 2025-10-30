import math

def count_powers(N):
    """
    Count the number of integers x between 1 and N, inclusive, 
    that can be expressed as x = a^b using some positive integer a and a positive integer b not less than 2.
    
    Parameters:
    N (int): The upper limit of the range.

    Returns:
    int: The number of integers that can be expressed as x = a^b.
    """
    # Initialize a set to store unique powers
    powers = set()

    # Check all possible values of a
    for a in range(1, int(math.sqrt(N)) + 1):
        # Check all possible values of b
        b = 2
        while a ** b <= N:
            # Add the power to the set
            powers.add(a ** b)
            b += 1

    # Return the number of unique powers
    return len(powers)

# Read the input from stdin
N = int(input())

# Print the result to stdout
print(count_powers(N))