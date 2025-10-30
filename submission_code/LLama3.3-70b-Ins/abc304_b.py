import sys

def approximate_number(N):
    """
    Approximate the given number N according to the specified rules.
    
    Parameters:
    N (int): The input number.
    
    Returns:
    int: The approximated number.
    """
    # Convert the number to a string to easily manipulate its digits
    str_N = str(N)
    
    # Determine the length of the number (i.e., the number of digits)
    length = len(str_N)
    
    # If N is less than or equal to 10^3-1, print N as it is
    if length <= 3:
        return N
    
    # If N is between 10^3 and 10^4-1, inclusive, truncate the ones digit of N
    elif length == 4:
        return int(str_N[:-1] + '0')
    
    # If N is between 10^4 and 10^5-1, inclusive, truncate the tens digit and all digits below it of N
    elif length == 5:
        return int(str_N[:-2] + '00')
    
    # If N is between 10^5 and 10^6-1, inclusive, truncate the hundreds digit and all digits below it of N
    elif length == 6:
        return int(str_N[:-3] + '000')
    
    # If N is between 10^6 and 10^7-1, inclusive, truncate the thousands digit and all digits below it of N
    elif length == 7:
        return int(str_N[:-4] + '0000')
    
    # If N is between 10^7 and 10^8-1, inclusive, truncate the ten-thousands digit and all digits below it of N
    elif length == 8:
        return int(str_N[:-5] + '00000')
    
    # If N is between 10^8 and 10^9-1, inclusive, truncate the hundred-thousands digit and all digits below it of N
    elif length == 9:
        return int(str_N[:-6] + '000000')

# Read the input from stdin
N = int(sys.stdin.readline().strip())

# Approximate the number and print the result
print(approximate_number(N))