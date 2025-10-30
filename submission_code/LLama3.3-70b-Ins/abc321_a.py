def is_321_like_number(n):
    """
    Checks if a given number is a 321-like Number.

    A 321-like Number is a positive integer where the digits are strictly decreasing from top to bottom.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is a 321-like Number, False otherwise.
    """
    # Convert the number to a string to easily access each digit
    str_n = str(n)

    # Iterate over each digit in the number
    for i in range(len(str_n) - 1):
        # If the current digit is not greater than the next digit, return False
        if int(str_n[i]) <= int(str_n[i + 1]):
            return False

    # If we've checked all digits and haven't returned False, the number is a 321-like Number
    return True


# Read the input from stdin
n = int(input())

# Check if the number is a 321-like Number and print the result
if is_321_like_number(n):
    print("Yes")
else:
    print("No")