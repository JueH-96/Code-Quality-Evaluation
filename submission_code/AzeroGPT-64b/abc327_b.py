def calculate_integer_root(B):
    """
    Finds a positive integer A such that A^A = B, or returns -1 if no such A exists.
    """
    # Since the exponentiation is A^A, the maximum possible value of A is 61 because 62^62 exceeds the long integer range.
    upper_limit = 62
    for A in range(1, upper_limit):
        if A**A == B:
            return A
    return -1

# Reading input from STDIN
import sys
B = int(input().strip())

# Calculating the result
result = calculate_integer_root(B)

# Printing the result to STDOUT
print(result)