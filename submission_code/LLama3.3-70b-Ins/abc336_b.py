def count_trailing_zeros(n):
    """
    This function calculates the number of trailing zeros in the binary representation of a given number.
    
    Parameters:
    n (int): The input number.
    
    Returns:
    int: The number of trailing zeros in the binary representation of n.
    """
    # Convert the number to binary and remove the '0b' prefix
    binary = bin(n)[2:]
    
    # Find the last occurrence of '1' in the binary string
    last_one_index = binary.rfind('1')
    
    # If '1' is not found, the entire string is zeros
    if last_one_index == -1:
        return len(binary)
    
    # The number of trailing zeros is the difference between the length of the string and the last index of '1' plus one
    return len(binary) - last_one_index - 1

# Read the input from stdin
n = int(input())

# Calculate and print the result
print(count_trailing_zeros(n))